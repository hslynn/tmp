from dolfin import *
import numpy as np

def set_vertex_value(u_list, coord, value_list):
    V = u_list[0].function_space()
    mesh = V.mesh()
    dofmap = V.dofmap()
    nvertices = mesh.ufl_cell().num_vertices()

    # Set up a vertex_2_dof list
    indices = [dofmap.tabulate_entity_dofs(1, 0)[i] for i in range(nvertices)]

    vertex_2_dof = dict()
    [vertex_2_dof.update(dict(vd for vd in zip(cell.entities(0), dofmap.cell_dofs(cell.index())[indices])))
            for cell in cells(mesh)]

    # Get the vertex coordinates
    X = mesh.coordinates()

    # Find the matching vertex (if it exists)
    vertex_idx = np.where((X == coord))[0] 
    assert vertex_idx, "No matching vertex!"
    vertex_idx = vertex_idx[0]
    dof_idx = vertex_2_dof[vertex_idx]
        
    for idx in range(len(u_list)):
        u_list[idx].vector()[dof_idx] = value_list[idx]

def set_local_max(local_max_pairs):
    for pair in local_max_paris:
        fs_u = pair[0].function_space()
        fs_local_max_u = pair[1].function_space()
        assert fs_u.mesh() == fs_local_max_u, 'u and local_max_u must have same mesh'
        mesh = fs_u.mesh()
        dofmap_u = fs_u.dofmap()
        dofmap_local_max_u = fs_local_max_u.dof_map() 
        for cell in cells(mesh):
            pair[1].vector()[dofmap.cell_dofs(cell.index())] = pair[0].vector()[dofmap.cell_dofs(cell.index())].max()
