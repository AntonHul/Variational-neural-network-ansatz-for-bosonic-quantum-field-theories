import jax
import jax.numpy as jnp
from flax import linen as nn
from src.embeddings_tf import embeddings
from src.jastrow_factors import log_jastrow

def log_cosh(x):
    """
    Computes the natural logarithm of cosh(x) in a numerically stable way.
    
    Approximation:
      log(cosh(x)) = |x| - log(2) + log(1 + exp(-2|x|))
    """
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def self_attention(q, k, v, mask=None):
    """
    Computes the scaled dot-product attention.
    
    Parameters:
      q: Query tensor.
      k: Key tensor.
      v: Value tensor.
      mask: Optional boolean mask (True = keep, False = mask out).

    Returns:
      The weighted sum of values based on query-key attention scores.
    """
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
    mask = jnp.matmul(mask, jnp.swapaxes(mask, -2, -1))
    if mask is not None:
        attn_logits = jnp.where(mask, attn_logits, -9e15)
    return jnp.matmul(nn.softmax(attn_logits, axis=-1), v)


class MultiheadAttention(nn.Module):
    """
    Multi-head attention module.
    

    Splits the input into multiple heads, performs attention on each, 
    and concatenates the results.

    Attributes:
      embed_dim: Total dimension of the model.
      head_dim: Dimension of each attention head.
    """
    embed_dim : int
    head_dim : int

    def setup(self):
        self.qkv_proj = nn.Dense(3*self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim)

    def __call__(self, x, mask=None):
        seq_length, embed_dim = x.shape
        assert embed_dim % self.head_dim == 0
        num_heads = embed_dim // self.head_dim
        qkv = self.qkv_proj(x).reshape(seq_length, num_heads, -1).transpose(1,0,2)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)
        values = self_attention(q, k, v, mask=mask).transpose(1,0,2).reshape(seq_length, embed_dim)
        return self.out_proj(values)


class EncoderBlock(nn.Module):
    """
    Transformer encoder block.
    
    Consists of a multi-head self-attention layer followed by a position-wise 
    feed-forward network, with residual connections and layer normalization.

    Attributes:
      embed_dim: The dimensionality of the input and output.
      head_dim: The dimensionality of each attention head.
      dim_feedforward: The dimensionality of the inner layer in the feed-forward network.
    """
    embed_dim : int
    head_dim : int
    dim_feedforward : int

    def setup(self):
        self.self_attn = MultiheadAttention(embed_dim=self.embed_dim, head_dim=self.head_dim)
        self.linear = [nn.Dense(self.dim_feedforward), log_cosh, nn.Dense(self.embed_dim)]
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, mask=None):
        x = x + self.self_attn(self.norm1(x), mask=mask)
        linear_out = self.norm2(x)
        for l in self.linear:
            linear_out = l(linear_out)
        return x + linear_out


class TransformerNQFS(nn.Module):
    """
    Transformer-based (TF) Neural Quantum Field State (NQFS) ansatz.
    
    This model represents the log-amplitude of the wavefunction: log(Psi(x)).
    It uses a Transformer encoder to process particle coordinates, followed by
    physics-informed terms (Jastrow factor, particle number regularization, boundary conditions).

    Attributes:
      num_layers: Number of Transformer encoder layers.
      embed_dim: Embedding dimension.
      head_dim: Dimension per attention head.
      dim_feedforward: Hidden dimension of the MLP in encoder blocks.
      dim_out: Output dimension before final projection.
      L: System length.
      periodic: Whether to enforce periodic boundary conditions.
      phys_dim: Physical dimension (1D, 2D).
      m: Particle mass.
      g: Interaction strength.
      jastrow_type: Type of Jastrow factor to include ('CS1D', 'LL1D', etc.).
      embed_type: Type of coordinate embedding ('Gaussian' or 'Periodic').
      k: Parameter for the embedding (e.g., width or frequency).
    """
    num_layers: int = 1
    embed_dim: int = 128
    head_dim: int = 32
    dim_feedforward: int = 512
    dim_out: int = 512
    L: float = 1.0
    periodic: bool = False
    phys_dim: int = 1
    m: float = 1.0
    g: float = 1.0
    jastrow_type: str = None
    embed_type: str = "Gaussian"
    k: float = 5.0
    
    # Configurable initial parameters
    q_n_mean_init: float = 5.0
    q_n_inv_softplus_width_init: float = 3.0
    q_n_inv_softplus_slope_init: float = 1.0

    def setup(self):
        # Encoder layers
        dim_in = self.embed_dim*2*self.phys_dim if self.embed_type=="Periodic" else self.embed_dim**self.phys_dim
        self.layers = [EncoderBlock(dim_in, self.head_dim, self.dim_feedforward) for _ in range(self.num_layers)]

        # Particle-number parameters
        self.q_n_mean = self.param('q_n_mean', nn.initializers.constant(self.q_n_mean_init), (1,))
        self.q_n_inv_softplus_width = self.param('q_n_inv_softplus_width', nn.initializers.constant(self.q_n_inv_softplus_width_init), (1,))
        self.q_n_inv_softplus_slope = self.param('q_n_inv_softplus_slope', nn.initializers.constant(self.q_n_inv_softplus_slope_init), (1,))
        self.norm = nn.LayerNorm()
        
    @nn.compact
    def __call__(self, x, mask_valid):
        mask = jax.lax.stop_gradient(mask_valid[:, None])
        n = jnp.sum(mask_valid)

        # Embedding
        x_emb = embeddings(self.embed_type)(x, self.L, self.embed_dim, self.phys_dim, self.k, self.periodic)
        for l in self.layers:
            x_emb = l(x_emb, mask=mask)
        x_emb = self.norm(x_emb)

        # Pooling
        x_emb = jax.scipy.special.logsumexp(x_emb, axis=0, where=mask).squeeze()
        x_emb = log_cosh(nn.Dense(features=self.dim_out)(x_emb))
        x_emb = nn.Dense(features=1)(x_emb)
        val = x_emb

        # L^(-n/2) contribution
        val += - n * self.phys_dim / 2 * jnp.log(self.L)

        # Jastrow contribution
        if self.jastrow_type is not None:
            val += (n >= 2) * log_jastrow(self.jastrow_type)(x, mask_valid, self.L, self.m, self.g, self.periodic)

        # q_n contribution
        val += 0.5 * self.log_q_n(n)

        # Cutoff factor for non-periodic systems
        if not self.periodic:
            val += self.log_cutoff_factor(x, n, mask_valid)
        return val.squeeze()


    def log_q_n(self, n):
        """
        Computes the log-probability factor for the particle number distribution.
        
        This implements a smoothed "box" or "window" function centered at `q_n_mean`
        with a learnable width and slope, penalizing particle numbers outside the preferred range.
        """
        q_n_width = jnp.log1p(jnp.exp(self.q_n_inv_softplus_width))
        c_1 = (2*self.q_n_mean - q_n_width)/2
        c_2 = (2*self.q_n_mean + q_n_width)/2
        s = jnp.log1p(jnp.exp(self.q_n_inv_softplus_slope))
        val = -jnp.log1p(jnp.exp(-s*(n-c_1))) - jnp.log1p(jnp.exp(s*(n-c_2)))
        return val


    def log_cutoff_factor(self, x, n, mask_valid):
        """
        Computes the boundary cutoff factor for open boundary conditions (Hard Wall).
        
        Forces the wavefunction to zero at the boundaries (0 and L) by adding
        log(x/L * (1 - x/L)), which goes to -infinity as x->0 or x->L.
        """
        val = (x % self.L)/self.L * (1 - (x % self.L)/self.L)
        val = jnp.sum(jnp.where(mask_valid[:, None], jnp.log(val+1e-16), 0))
        val -= n * jnp.log(self.L/30)
        return val
