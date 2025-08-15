def inspect_configs(configs):
    """
    Prints useful information about a set of ase.Atoms objects

    :param configs: list of ase.Atoms objects to inspect
    :type configs: list[ase.Atoms]
    """
    elements = set()
    properties = {}
    atoms = 0
    for c in configs:
        atoms += len(c)
        elems = c.get_chemical_symbols()
        for e in elems:
            elements.add(e)
        info_keys = c.info.keys()
        for key in info_keys:
            if key not in ['po-id', 'co-id', 'ds-id']:
                if key in properties:
                    properties[key] += 1
                else:
                    properties[key] = 1
        array_keys = c.arrays.keys()
        for key in array_keys:
            if key not in ['numbers', 'positions']:
                if key in properties:
                    properties[key] += 1
                else:
                    properties[key] = 1

    print_statement = f"There are {len(configs)} configurations" \
                      f" totaling {atoms} atoms. \n"

    for k, v in properties.items():
        strng = f"{v} contain the key {k}. \n"
        print_statement += strng

    print(print_statement)
