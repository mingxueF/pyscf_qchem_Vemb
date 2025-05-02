#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob

def zr_frag(fname, separator="----", fmt="string"):
    """ Get fragment coordinates from .zr file 
    
    Parameters
    ----------
    fname: string
        ZR filename
    separator: string
        Separator.
    fmt: string
        Format specifier. Possible options: "list", "string"
        
    Returns
    -------
    frags: dict
        Dictionary of list of lists with fragment coordinates and atom symbol,
        e.g. ``frags["A"] = ["C 0.0 0.1 0.0", "..."]``. If no separator is
        found (normal xyz file), only one fragment is assumed ("AB").
        
        If fragments are found ``frags`` will contain the keys 'A', 'B', 'AB',
        'AB_ghost'.
    
    
    """
    if fmt.lower() not in ("string", "list"):
        raise ValueError("Invalid format option specified! Use either 'string' or 'list'.")
    with open(fname) as zr:
        rl = zr.readlines()
    line_B = 0
    frags = {}
    for i, line in enumerate(rl):
        if separator in line:
            line_B = i
    
    if line_B == 0:
        frags["AB"] = list(map(str.split, rl[0:]))
    else:
        frags["A"] = list(map(str.split, rl[0:line_B]))
        frags["B"] = list(map(str.split, rl[line_B+1:]))
        frags["AB_ghost"] = frags["A"] + list(map(lambda x: ["@"+x[0]]+x[1:],
                                                  frags["B"]))
        frags["BA_ghost"] = list(map(lambda x: ["@"+x[0]]+x[1:], frags["A"]))+\
                            frags["B"]
        frags["AB"] = frags["A"] + frags["B"]

    if fmt=="list":
        return frags
    elif fmt=="string":
        for key in frags:
#            frags[key] = "\n".join(["\t".join(s) for s in frags[key]])
#            "{0:<8}{1:<16}{2:<16}{3:<16}".format(*s)
            frags[key] = "\n".join(["    ".join(s) for s in frags[key]])
        return frags

def find_file(directory, extension="in"):
    """ Find input file in a directory.
    
    Returns
    -------
     : str or int
        Path to file or 0 in case of AssertionError.
    """
    dir_list = [fn for fn in glob.glob(directory+"/*."+extension)]
    try:
        assert len(dir_list) == 1
        return dir_list[0]
    except AssertionError:
        print("AssertionError: Could not determine single "+ extension +
              "file in " + directory + " !")
        return 0
