import jax
import jax.numpy as jnp
import jax.lax as lax

from jaxpm.kernels import fftk, cic_compensation

def cic_paint(mesh, positions, weight=None):
  """ Paints positions onto mesh
  mesh: [nx, ny, nz]
  positions: [npart, 3]
  """
  positions = jnp.expand_dims(positions, 1)
  floor = jnp.floor(positions)
  connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], 
                           [0., 0, 1], [1., 1, 0], [1., 0, 1], 
                           [0., 1, 1], [1., 1, 1]]])

  neighboor_coords = floor + connection
  kernel = 1. - jnp.abs(positions - neighboor_coords)
  kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]  
  if weight is not None:
        kernel = jnp.multiply(jnp.expand_dims(weight, axis=-1), kernel)
  
  neighboor_coords = jnp.mod(neighboor_coords.reshape([-1,8,3]).astype('int32'), jnp.array(mesh.shape))

  dnums = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0, 1, 2),
    scatter_dims_to_operand_dims=(0, 1, 2))
  mesh = lax.scatter_add(mesh, 
                         neighboor_coords, 
                         kernel.reshape([-1,8]),
                         dnums)
  return mesh

def cic_read(mesh, positions):
  """ Paints positions onto mesh
  mesh: [nx, ny, nz]
  positions: [npart, 3]
  """  
  positions = jnp.expand_dims(positions, 1)
  floor = jnp.floor(positions)
  connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], 
                           [0., 0, 1], [1., 1, 0], [1., 0, 1], 
                           [0., 1, 1], [1., 1, 1]]])

  neighboor_coords = floor + connection
  kernel = 1. - jnp.abs(positions - neighboor_coords)
  kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]  

  neighboor_coords = jnp.mod(neighboor_coords.astype('int32'), jnp.array(mesh.shape))

  return (mesh[neighboor_coords[...,0], 
               neighboor_coords[...,1], 
               neighboor_coords[...,3]]*kernel).sum(axis=-1)

def cic_paint_2d(mesh, positions, weight):
  """ Paints positions onto a 2d mesh
  mesh: [nx, ny]
  positions: [npart, 2]
  weight: [npart]
  """
  positions = jnp.expand_dims(positions, 1)
  floor = jnp.floor(positions)
  connection = jnp.array([[0, 0], [1., 0], [0., 1], [1., 1]])

  neighboor_coords = floor + connection
  kernel = 1. - jnp.abs(positions - neighboor_coords)
  kernel = kernel[..., 0] * kernel[..., 1] 
  if weight is not None:
    kernel = kernel * weight[...,jnp.newaxis]
  
  neighboor_coords = jnp.mod(neighboor_coords.reshape([-1,4,2]).astype('int32'), jnp.array(mesh.shape))

  dnums = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0, 1),
    scatter_dims_to_operand_dims=(0, 1))
  mesh = lax.scatter_add(mesh, 
                         neighboor_coords, 
                         kernel.reshape([-1,4]),
                         dnums)
  return mesh

def compensate_cic(field):
  """
  Compensate for CiC painting
  Args:
    field: input 3D cic-painted field
  Returns:
    compensated_field
  """
  nc = field.shape
  kvec = fftk(nc)

  delta_k = jnp.fft.rfftn(field)
  delta_k = cic_compensation(kvec) * delta_k
  return jnp.fft.irfftn(delta_k)


def compute_attention(query, key):
    return jax.nn.softmax(jnp.dot(query, key.T))

def attention_read(mesh, positions, embed_model, params,): 
    positions = jnp.expand_dims(positions,1)
    floor = jnp.floor(positions)
    connection = jnp.array(
        [
            [[0, 0, 0], [1., 0, 0], [0., 1, 0], 
            [0., 0, 1], [1., 1, 0], [1., 0, 1], 
            [0., 1, 1], [1., 1, 1]]
        ]
    )
    neighbour_coords = floor + connection
    distances = positions - neighbour_coords
    neighbour_coords = jnp.mod(neighbour_coords.astype('int32'), jnp.array(mesh.shape))
    neighbour_mesh = mesh[
        neighbour_coords[...,0], 
        neighbour_coords[...,1], 
        neighbour_coords[...,3],
    ]
    query = jax.vmap(embed_model.apply, in_axes=(None, 0))(params['distances'], distances)
    key = jax.vmap(embed_model.apply, in_axes=(None, 0))(params['mesh'], neighbour_mesh[..., None])
    
    # Using vmap for compute_attention and dot product
    attention_weights = jax.vmap(compute_attention)(query, key)
    dot_product = jax.vmap(jnp.dot)(attention_weights, neighbour_mesh)
    
    return jnp.sum(dot_product, axis=-1)