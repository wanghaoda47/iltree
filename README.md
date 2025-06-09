# iltree
A modified python code for merger trees of Illustris.

This python code is a modified version of illustris_python.sublink.

Changes:
  1. The 'loadTree' function can accept a list of ids and return a list of trees now. When a single tree is need, one can extract the tree from this list with 'extractTree'.
  2. The functions 'numMergers' and 'maxPastMass' of the official illustris_python is changed into 'numMergersType' and 'maxPastMass'. Functions 'numMergers' and 'maxPastMass' now process the total mass of subhalos instead of mass of certain type of particles.
  3. The determination of mass ratio can be bypassed by setting the parameter 'minMassRatio' as 0. When one adopt this change, the result will contain all mergers regardless of the mass ratio of subhalos involved. This change can also result in a much faster calculation speed.
  4. New function 'MergerIDs' is provided to find all the index of mergers in a tree. The result appears as such a list: [[a0,b0,c0],[a1,b1,c1],...]. Each set of a,b,c refer to the index of the First Progenitor, the Secondary Progenitor, the Descendent in the tree list. When a Merger involves more than two halos, a few sets of a,b,c is provided, where a and c do not change and b changes to the next progenitor.
  5. Function 'Mergers' returns more than 'MergerIDs'. Fields can be selected by input.
  6. Functions 'MergerIDsType' and 'MergersType' select mergers by mass ratio of a certain type of particles rather than the total mass of subhalos.
