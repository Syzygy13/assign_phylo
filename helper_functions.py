"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from Bio.Align import substitution_matrices

blosum62 = substitution_matrices.load("BLOSUM62")

class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    
    # Initialise distance matrix and node list
    distance_matrix = np.array(distances, dtype=float)
    node_list = []
    for label in labels:
        node_list.append(Node(str(label), None, 0.0, None, 0.0))
    node_count = 0

    if len(node_list) == 1:
        return node_list[0]

    # Stop at 2 nodes remaining to make root
    while len(node_list) > 2:
        num = len(node_list)

        # Sum of distances from each node to all others
        row_sums = distance_matrix.sum(axis=1)

        # Q_ij = (n - 2) * D_ij - sum_n_k(D_ik) - sum_n_k(D_jk)
        # Use inf instead of 0 to help finding the minimum value in Q matrix
        q_matrix = (num - 2) * distance_matrix - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(q_matrix, np.inf)

        # Smallest value in Q value's index is the pair to combine
        i, j = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)

        pair_distance = distance_matrix[i, j]

        # Distance from each joined node to the new parent node:
        # Delta_iu = D_ij / 2 + (sum_n_k(D_ik) - sum_n_k(D_jk)) / (2 * (n - 2))
        first_branch_length = 0.5 * pair_distance + (row_sums[i] - row_sums[j]) / (2 * (num - 2))
        second_branch_length = pair_distance - first_branch_length

        # Create the combined node and increase node count for naming
        new_node = Node(
            f"{node_count}",
            node_list[i],
            float(first_branch_length),
            node_list[j],
            float(second_branch_length),
        )
        node_count += 1

        # Distance from the new node to every other node:
        # D_uk = (D_ik + D_jk - D_ij) / 2
        remaining_indices = []
        for index in range(num):
            if index not in (i, j):
                remaining_indices.append(index)

        distances_to_new_node = []
        for index in remaining_indices:
            distance_to_new_node = 0.5 * (
                distance_matrix[i, index] + distance_matrix[j, index] - pair_distance
            )
            distances_to_new_node.append(distance_to_new_node)

        # New matrix of zeroes then updated distances and append the new node at the end
        remove_combined_matrix = distance_matrix[np.ix_(remaining_indices, remaining_indices)]
        updated_matrix = np.zeros((num - 1, num - 1))
        updated_matrix[:-1, :-1] = remove_combined_matrix
        updated_matrix[-1, :-1] = distances_to_new_node
        updated_matrix[:-1, -1] = distances_to_new_node

        distance_matrix = updated_matrix
        node_list = [node_list[index] for index in remaining_indices]
        node_list.append(new_node)

    # Join the last two nodes under a root node
    root_distance = float(distance_matrix[0, 1]) / 2
    return Node(
        "ROOT",
        node_list[0],
        root_distance,
        node_list[1],
        root_distance,
    )


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Use an array as ints would be locally updated and not persist across recursive calls
    next_leaf_row = [0]
    # Use recursion to draw the tree. Create a node if there are no children otherwise traverse left then right
    def draw_subtree(node: Node, x: float) -> float:
        # Create the leaf node and return the y position on the graph
        if node.left is None and node.right is None:
            y = next_leaf_row[0]
            next_leaf_row[0] += 1
            ax.annotate(node.name, (x, y))
            return y

        # Recursive calls to increase length of the branches and draw nodes
        left_dist = x + node.left_distance
        right_dist = x + node.right_distance
        left_y = draw_subtree(node.left, left_dist)
        right_y = draw_subtree(node.right, right_dist)

        # Vertical connector between the children, then a horizontal branch to each
        ax.plot([x, x], [left_y, right_y], color="red", linewidth=1)
        ax.plot([x, left_dist], [left_y, left_y], color="green", linewidth=1)
        ax.plot([x, right_dist], [right_y, right_y], color="purple", linewidth=1)

        # Root y-value
        return (left_y + right_y) / 2

    # Draw tree and get the height of the root node
    root_y = draw_subtree(tree, 0.0)

    # Short point for the root on the left of plot
    ax.plot([-0.5, 0], [root_y, root_y], color="black", linewidth=1)

    # General plot information
    ax.set_yticks(list(range(next_leaf_row[0])))
    ax.set_xlabel("Genetic distance")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    # Loop to go through every node. Parents are stored prior to their children
    nodes_in_order = []
    nodes_to_visit = [tree]
    while len(nodes_to_visit) > 0:
        current_node = nodes_to_visit.pop()
        nodes_in_order.append(current_node)

        if current_node.left is not None:
            nodes_to_visit.append(current_node.left)
        if current_node.right is not None:
            nodes_to_visit.append(current_node.right)

    # Look through node list and find number of children each node has
    num_children = {}
    for current_node in reversed(nodes_in_order):

        # If the node is a leaf then it only counts itself. Done prior to parent of leaves
        if current_node.left is None and current_node.right is None:
            num_children[current_node] = 1
            continue

        # Parent nodes have value equal to the number of children of their children
        left_count = num_children[current_node.left]
        right_count = num_children[current_node.right]
        num_children[current_node] = left_count + right_count

        # Reorganise the tree based on number of children. Fewer children is further down on the plot
        if left_count > right_count:
            temp_node = current_node.left
            current_node.left = current_node.right
            current_node.right = temp_node

            temp_distance = current_node.left_distance
            current_node.left_distance = current_node.right_distance
            current_node.right_distance = temp_distance


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()

def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """
    # Initialise lengths of sequences and matrices
    n, m = len(seq1), len(seq2)
    scores = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    arrows = [[None for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + scoring_function(seq1[i - 1], "-")
        arrows[i][0] = "up"

    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + scoring_function("-", seq2[j - 1])
        arrows[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Fill in scores matrix with scores using Needleman-Wunsch algorithm
            match = scores[i - 1][j - 1] + scoring_function(seq1[i - 1], seq2[j - 1])
            up = scores[i - 1][j] + scoring_function(seq1[i - 1], "-")
            left = scores[i][j - 1] + scoring_function("-", seq2[j - 1])
            scores[i][j] = max(match, up, left)

            # Fill in arrows matrix with directions using results from algorithm
            if scores[i][j] == match:
                arrows[i][j] = "diagonal"
            elif scores[i][j] == up:
                arrows[i][j] = "up"
            else:
                arrows[i][j] = "left"

    aligned_seq1 = []
    aligned_seq2 = []
    i, j = n, m

    # Find full path back to start of matrix to get aligned sequences
    while i > 0 or j > 0:
        if arrows[i][j] == "diagonal":
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif arrows[i][j] == "up":
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    # Reverse directions to get aligned sequences
    aligned_seq1.reverse()
    aligned_seq2.reverse()

    aligned_seq1_combined = "".join(aligned_seq1)
    aligned_seq2_combined = "".join(aligned_seq2)

    return aligned_seq1_combined, aligned_seq2_combined, float(scores[n][m])

def scoring_function(aa_i,aa_j):
    gap_penalty = -8

    if aa_i == "-" or aa_j == "-":
        score = gap_penalty
    else:
        score = blosum62[aa_i, aa_j]

    return (score)
