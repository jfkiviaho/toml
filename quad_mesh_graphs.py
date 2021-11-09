from tacs import TACS

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpi4py import MPI
import functools
import operator

def convertTacsConnectivity(data, ptr):
    '''
    Convert TACS layout of element-to-vertex connectivity to list of lists

    Parameters
    ----------
    data : numpy.ndarray
        array of non-negative integers corresponding to vertex IDs
    ptr : numpy.ndarray
        array of non-negatives integers corresponding to indices in first array

    Returns
    -------
    elem_to_vertex : list
        list of lists of vertex IDs describing element-to-vertex connectivity

    Notes
    -----
    In order to support different types of elements---different numbers of
    vertices per element---in the same mesh, TACS internally represents the
    element-to-vertex connectivity using two arrays. The first is a "data"
    array that simply contains a list of non-negative integers that identify
    vertices in a global numbering of the vertices. The second is a "pointer"
    array that contains a set of non-negative integers that identify indices in
    the first array; a pair of adjacent indices in the pointer array defines a
    "slice" into the first array, and that slice contains the subset of vertex
    IDs that correspond to a single element. Since a list data structure in
    Python allows for members of the list to be of different size, we might as
    well consolidate the two arrays TACS provides into a list of lists.
    '''
    return list(map(lambda a: a.tolist(), np.array_split(data, ptr[1:-1])))

def fixTacsConnectivityForGmshQuad4Mesh(quad_to_vert):
    '''
    Fix the connectivity that TACS reads from a Gmsh-produced BDF file for a
    QUAD4 mesh to account for the fact that Gmsh writes the connectivity for a
    QUAD4 in a non-standard way, causing TACS to read it incorrectly
    '''
    # Swap the last two verts
    for quad in quad_to_vert:
        quad[2], quad[3] = quad[3], quad[2]

    return quad_to_vert

def reconstructQuad4EdgeToVertConn(quad_to_vert):
    '''
    For a mesh made exlusively of QUAD4 elements, determine the edge-to-vertex
    from the given element-to-vertex connectivity and return

    Parameters
    ----------
    quad_to_vert : list
        list of lists of vertex IDs describing element-to-vertex connectivity

    Returns
    -------
    edge_to_vert : list
        list of lists of vertex IDs describing edge-to-vertex connectivity
    '''
    # Break the lists of lists of vertices in each quad into a list of lists of
    # tuples of vertices in each edge in each quad
    def compose(f, g):
        ''' Composition of a pair functions '''
        return lambda x: f(g(x))

    def separateQuad4IntoEdges(q):
        ''' QUAD4 can be decomposed into edges in standard way '''
        return [[q[0], q[1]], [q[1], q[2]], [q[2], q[3]], [q[3], q[0]]]

    def sortAndTuplify(edges):
        ''' Sorting and conversion to tuple needed for later hashing '''
        return list(map(lambda e: tuple(sorted(e)), edges))

    quad_to_edge_to_vert = list(
            map(
                compose(sortAndTuplify, separateQuad4IntoEdges), 
                quad_to_vert
            )
        )

    # Concatenate the list of lists of tuples to obtain a non-unique list of
    # tuples of vertices
    non_unique_edge_to_vert = functools.reduce(
            operator.add, 
            quad_to_edge_to_vert
        )

    # Filter the non-unique tuples of vertices (edges) out
    edge_to_vert = list(set(non_unique_edge_to_vert))
    

    # Convert the edge tuples back to lists for PyTorch Geometric compatibility
    edge_to_vert = list(map(lambda e: list(e), edge_to_vert))

    return edge_to_vert

def reconstructQuad4ElemToElemConn(quad_to_vert):
    '''
    For a mesh made exlusively of QUAD4 elements, determine the
    element-to_element connectivity from the given element-to-vertex
    connectivity and return

    Parameters
    ----------
    quad_to_vert : list
        list of lists of vertex IDs describing element-to-vertex connectivity

    Returns
    -------
    quad_to_quad : list
        list of lists of vertex IDs describing element-to-element connectivity
    '''
    # Find maximum vertex ID referenced in quad-to-vertex connectivity
    num_verts = functools.reduce(max, map(max, quad_to_vert)) + 1

    # Invert quad-to-vertex connectivity
    vert_to_quad = [[] for _ in range(num_verts)]
    for quad, verts in enumerate(quad_to_vert):
        for vert in verts:
            vert_to_quad[vert].append(quad)

    # Combine quad_to_vertex and vertex 
    num_quads = len(quad_to_vert)
    quad_to_quad = [set() for _ in range(num_quads)]
    for quad, verts in enumerate(quad_to_vert):
        for vert in verts:
            quads = vert_to_quad[vert]
            quad_to_quad[quad].update(quads)

        # Remove self-reference
        quad_to_quad[quad].remove(quad)

    # Convert the quad sets into lists
    quad_to_quad = list(map(lambda qs: list(qs), quad_to_quad))

    return quad_to_quad

def convertVertToVertIntoEdgeToVert(vert_to_vert):
    '''
    For a given vertex-to-vertex connectivity, determine the edge-to-vertex
    connectivity

    Parameters
    ----------
    vert_to_vert : list
        list of lists of vertex IDs describing vertex-to-vertex connectivity

    Returns
    -------
    edge_to_vert : list
        list of lists of vertex IDs describing edge-to-vertex connectivity
    '''
    edge_to_vert = set()
    for vert, connected_verts in enumerate(vert_to_vert):
        for connected_vert in connected_verts:
            edge = tuple(sorted([vert, connected_vert]))
            edge_to_vert.add(edge)

    # Convert set of edges into a list and convert tuples of vertices into
    # lists for compatibility with PyTorch Geometric
    edge_to_vert = list(map(lambda e: list(e), edge_to_vert))

    return edge_to_vert

def plotQuad4MeshGraphAndDualGraph(vert_coords, quad_to_vert, edge_to_vert, 
        dual_edge_to_vert):
    '''
    '''
    fig, ax = plt.subplots(figsize=(10,8))

    # Reshape the vertex coordinate array for easier indexing
    vert_coords = vert_coords.reshape((-1,3))

    # Extract mesh graph edge coordinates
    graph_edge_coords = []
    for edge in edge_to_vert:
        vert0, vert1 = edge[0], edge[1]
        graph_edge_coords.append(
                [
                    (vert_coords[vert0,0], vert_coords[vert0,1]),
                    (vert_coords[vert1,0], vert_coords[vert1,1])
                ]
            )

    graph_colors = np.array([0,0,1])
    graph_edges = mc.LineCollection(
            graph_edge_coords, 
            colors=graph_colors
        )

    # Extract mesh dual graph edge coordinates
    dual_graph_edge_coords = []
    for edge in dual_edge_to_vert:
        quad0, quad1 = edge[0], edge[1]
        cent0_coords = 0.25*np.sum(vert_coords[quad_to_vert[quad0]], axis=0)
        cent1_coords = 0.25*np.sum(vert_coords[quad_to_vert[quad1]], axis=0)
        dual_graph_edge_coords.append(
                [
                    (cent0_coords[0], cent0_coords[1]),
                    (cent1_coords[0], cent1_coords[1]),
                ]
            )

    dual_graph_colors = np.array([1,0,0])
    dual_graph_edges = mc.LineCollection(
            dual_graph_edge_coords, 
            colors=dual_graph_colors
        )

    # Plot graph and dual graph
    ax.add_collection(graph_edges)

    ax.add_collection(dual_graph_edges)

    ax.autoscale()
    ax.margins(0.1)
    plt.show()

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    #filename = "decks/simple_quad_mesh.bdf"
    #filename = "decks/less_simple_quad_mesh.bdf"
    #filename = "decks/bracket_quad_mesh.bdf"
    filename =  "decks/short_cantilever_with_spcs.bdf"

    # Load mesh through TACS
    reader = TACS.MeshLoader(comm)
    reader.scanBDFFile(filename)

    # Extract element-to-vertex connectivity
    elem_to_vert_ptr, elem_to_vert_data, _, _ = reader.getConnectivity()

    # Convert TACS connecitivity format to list of lists
    elem_to_vert = convertTacsConnectivity(elem_to_vert_data, elem_to_vert_ptr)
    elem_to_vert = fixTacsConnectivityForGmshQuad4Mesh(elem_to_vert)

    # Assuming all elements are QUAD4, reconstruct edge-to-vertex connectivity
    edge_to_vert = reconstructQuad4EdgeToVertConn(elem_to_vert)

    # Assuming all elements are QUAD4, reconstruct element-to-element
    # connectivity
    quad_to_quad = reconstructQuad4ElemToElemConn(elem_to_vert)

    # Convert the quad-to-quad connectivity into an edge-to-vertex
    # connectivity, where the quads themselves are considered vertices of the
    # graph
    dual_edge_to_vert = convertVertToVertIntoEdgeToVert(quad_to_quad)

    # Plot the QUAD4 mesh graph and the QUAD4 mesh dual graph
    _, _, _, vert_coords = reader.getConnectivity()
    plotQuad4MeshGraphAndDualGraph(vert_coords, elem_to_vert, edge_to_vert, 
        dual_edge_to_vert)
