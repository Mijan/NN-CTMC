import numpy as np

def getUniqueStoichiometryMapping(stoich):
    mapping = [None] * len(stoich)
    stoich_to_rct = {}
    unique_rct_nbr = 0

    for rct_nbr, stoich_vec in enumerate(stoich):
        stoich_tuple = tuple(stoich_vec)

        if stoich_tuple not in stoich_to_rct:
            stoich_to_rct[stoich_tuple] = unique_rct_nbr
            unique_rct_nbr += 1

        mapping[rct_nbr] = stoich_to_rct[stoich_tuple]

    return mapping


def getReactionsForObservations(observation_vector, stoich):
    unique_reaction_mapping = None
    unique_stoich = np.unique(stoich)
    if(len(unique_stoich) != len(stoich)):
        unique_reaction_mapping = getUniqueStoichiometryMapping(stoich)

    if not unique_reaction_mapping:
        unique_reaction_mapping = [i for i in range(len(stoich))]

    differences = np.diff(observation_vector, axis=0)

    # Map stoichiometry to a list of its corresponding reaction numbers
    stoich_dict = {}
    for rct_nbr, stoich_vect in enumerate(stoich):
        key = tuple(stoich_vect)
        stoich_dict[key] = unique_reaction_mapping[rct_nbr]

    # Map differences to unique reactions using the mapping
    reactions = np.array([stoich_dict.get(tuple(diff), -1) for diff in differences])
    return reactions, unique_reaction_mapping