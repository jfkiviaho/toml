'''
Prepare Short Cantilever
------------------------
This script adds the boundary conditions (single point constraints or SPCs) to
the Short Cantilever mesh file (BDF), so that it can be ingested into TACS for
analysis and then topology optimization.

The Short Cantilever test case was prepared in Gmsh following the steps below.
See the Gmsh script file "half_short_cantilever.geo" for more specifics.
1. The upper half of rectangle was created, specifying a mesh size of 0.025 at
each point (corresponding to the approximate mesh size in the 80 x 40
structured grid used in the TONR paper).
2. The upper half was reflected along the x-axis, the longitudinal axis of the
beam, using the symmetry transformation provided by Gmsh.
3. The surface geometry created for the upper half and the surface geometry
generated through symmetry transformation for the lower half were grouped into
a "physical group" in Gmsh.
4. The left edges of both the upper and lower half were grouped into another
physical group in Gmsh.
5. The surfaces were meshed with the options provided in the file
"half_short_cantilever.geo.opt"
half of the geometry
6. The mesh was exported from Gmsh with the options "Long field" and
"Elementary entity" with "Save all elements" unchecked.

Steps 3, 4 and 6 are crucial parts of this procedure. If we check the box for
"save all elements", Gmsh will export all the elements generated by the mesher
(Step 5), including one-dimensional bar elements on the element edges. We don't
want that. If we leave the box unchecked (Step 6), Gmsh won't export any part
of the mesh at all, unless we have included the geometry that underlies that
part of the mesh in a physical group.  Obviously, we want the quad elements
generated on the surfaces to be part of the exported mesh file, so we incude
those in a physical group (Step 3). But we also want Gmsh to help us identify
which nodes lie on the boundary so that we can add constraints on them in this
post-processing script. So we add the clamped (left) edges to another physical
group (Step 4).
'''
import re
import subprocess

# When you specify the "long field" option, the BDF output from Gmsh breaks
# each line defining the grid points (mesh nodes) and continues it on the
# next line. To get this task done quicker, I'm not going to learn how to
# do regex over multiple lines; I'm just going to join the broken lines and
# rewrite the BDF to a temporary file
filename = 'decks/short_cantilever.bdf'
temp_filename = filename + '-tmp' 
out_filename = 'decks/short_cantilever_with_spcs.bdf'
with open(filename, 'r') as f:
    with open(temp_filename, 'w') as g:
        lines_in = f.readlines()
        lines_out = []
        for i, line_in in enumerate(lines_in):
            if line_in[0] == '*':
                line_before = lines_in[i-1]
                line_out = line_before[:-1] + line_in[1:]
                lines_out[-1] = line_out
            else:
                lines_out.append(line_in)

        for line_out in lines_out:
            g.write(line_out)

# Parse mesh node coordinates and boundary edge connectivity from temporary file
node_coords = []
bc_edge_to_vert = []
node_line_pattern = re.compile('^GRID.*$')
edge_line_pattern = re.compile('^CBAR.*$')
with open(temp_filename, 'r') as f:
    for line in f.readlines():
        if node_line_pattern.match(line):
            cols = re.split('\s+', line)
            x = float(cols[3])
            y = float(cols[4])
            z = float(cols[5])
            node_coords.append([x, y, z])
        elif edge_line_pattern.match(line):
            cols = re.split('\s+', line)
            edge = [int(cols[3]), int(cols[4])]
            bc_edge_to_vert.append(edge)

# Get the IDs of the nodes on the clamped boundary
bc_node_ids = set()
for edge in bc_edge_to_vert:
    bc_node_ids.update(edge)
bc_node_ids = list(bc_node_ids)

# Add the SPCs and fixed connectivity to the original file
quad_line_pattern = re.compile(
        '^CQUAD4\s*' 
        + '([0-9]*)\s*'  # match and capture element ID
        + '([0-9]*)\s*'  # match and capture the zone ID
        + '([0-9]*)\s*'  # match and capture the vert IDs...
        + '([0-9]*)\s*' 
        + '([0-9]*)\s*'
        + '([0-9]*)\s*'
        + '$'
    )
end_file_pattern = re.compile('^ENDDATA.*$')
quad_id = 1
with open(filename, 'r') as f:
    with open(out_filename, 'w') as g:
        for line in f.readlines():
            # If at the end of the file, add constraints
            if end_file_pattern.match(line):
                for node in bc_node_ids:
                    # TODO Hey Tingwei, this is where to add them, I think
                    pass

            # Don't copy the bar elements
            if edge_line_pattern.match(line):
                pass
            # Fix the quad connecitivity, replace zone info, restart numbering
            elif quad_line_pattern.match(line):
                quad_line_match = quad_line_pattern.match(line)
                cols = re.split('(\s+)', line)  # split but retain separators
                empty_string = cols.pop()
                last_filled = cols.pop()
                last_space, end_line_char = last_filled[:-1], last_filled[-1:]
                cols.extend([last_space, end_line_char, empty_string])

                # Fix the connectivity in zone 1, the non-reflected zone
                zone = int(cols[4])
                if zone == 1:
                    cols_cpy = cols.copy()

                    # Reverse the numbering of the vertices
                    cols[ 6] = cols_cpy[12]
                    cols[ 8] = cols_cpy[10]
                    cols[10] = cols_cpy[ 8]
                    cols[12] = cols_cpy[ 6]

                    # Swap the padding spaces too to preserve formatting
                    cols[ 7] = cols_cpy[13]
                    cols[ 9] = cols_cpy[11]
                    cols[11] = cols_cpy[ 9]
                    cols[13] = cols_cpy[ 7]

                # Renumber the quads
                old_num = cols[2]
                new_num = str(quad_id)
                pad_size = len(old_num) - len(new_num)
                new_num += ' '*pad_size
                cols[2] = new_num

                # Replace the zone informations with the element ID. TACS uses
                # the zone info to set design variables. Since there is a
                # design variables associated with each element, it makes sense
                # to set the zone info to the element ID
                cols[4] = new_num  # overwrite zone info
                cols[5] = cols[3]  # overwrite padding


                new_line = ''.join(cols)
                g.write(new_line)

                quad_id += 1

            else:
                # Otherwise, copy the line into the new file
                g.write(line)

# Delete the temporary file
del_cmd = [
        'rm',
        temp_filename
    ]
subprocess.run(del_cmd)
