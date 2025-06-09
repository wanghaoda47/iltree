"""
Author: Haoda Wang
Date: 9th June 2025
Modified version based on illustris_python.sublink, suitable for illustris public data.
"""

import numpy as np
import h5py
import glob
import six
import os
import illustris_python as il

from illustris_python.groupcat import gcPath, offsetPath
from illustris_python.util import partTypeNum


def treePath(basePath, treeName, chunkNum=0):
    """ Return absolute path to a SubLink HDF5 file (modify as needed). """
    # tree_path = '/trees/' + treeName + '/' + 'tree_extended.' + str(chunkNum) + '.hdf5'
    tree_path = os.path.join('trees', treeName, 'tree_extended.' + str(chunkNum) + '.hdf5')

    _path = os.path.join(basePath, tree_path)
    if len(glob.glob(_path)):
        return _path

    # new path scheme
    _path = os.path.join(basePath, os.path.pardir, 'postprocessing', tree_path)
    if len(glob.glob(_path)):
        return _path

    # try one or more alternative path schemes before failing
    _path = os.path.join(basePath, 'postprocessing', tree_path)
    if len(glob.glob(_path)):
        return _path

    raise ValueError("Could not construct treePath from basePath = '{}'".format(basePath))


def treeOffsets(basePath, snapNum, id, treeName):
    """ Handle offset loading for a SubLink merger tree cutout. """
    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum) or treeName == "SubLink_gal":
        # load groupcat chunk offsets from separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['FileOffsets/Subhalo'][()]

        offsetFile = offsetPath(basePath, snapNum)
        prefix = 'Subhalo/' + treeName + '/'

        groupOffset = id
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['Header'].attrs['FileOffsets_Subhalo']

        # calculate target groups file chunk which contains this id
        groupFileOffsets = int(id) - groupFileOffsets
        fileNum = np.max(np.where(groupFileOffsets >= 0))
        groupOffset = groupFileOffsets[fileNum]

        offsetFile = gcPath(basePath, snapNum, fileNum)
        prefix = 'Offsets/Subhalo_Sublink'

    with h5py.File(offsetFile, 'r') as f:
        # load the merger tree offsets of this subgroup
        RowNum     = f[prefix+'RowNum'][groupOffset]
        LastProgID = f[prefix+'LastProgenitorID'][groupOffset]
        SubhaloID  = f[prefix+'SubhaloID'][groupOffset]
        return RowNum, LastProgID, SubhaloID

offsetCache = dict()

def subLinkOffsets(basePath, treeName, cache=True):
    # create quick offset table for rows in the SubLink files
    if cache is True:
        cache = offsetCache

    if type(cache) is dict:
        path = os.path.join(basePath, treeName)
        try:
            return cache[path]
        except KeyError:
            pass

    search_path = treePath(basePath, treeName, '*')
    numTreeFiles = len(glob.glob(search_path))
    if numTreeFiles == 0:
        raise ValueError("No tree files found! for path '{}'".format(search_path))
    offsets = np.zeros(numTreeFiles, dtype='int64')

    for i in range(numTreeFiles-1):
        with h5py.File(treePath(basePath, treeName, i), 'r') as f:
            offsets[i+1] = offsets[i] + f['SubhaloID'].shape[0]

    if type(cache) is dict:
        cache[path] = offsets

    return offsets

def loadTree(basePath, snapNum, id, fields=None, onlyMPB=False, onlyMDB=False, treeName="SubLink", cache=True):
    """ Load portion of Sublink tree, for a given subhalo, in its existing flat format.
        (optionally restricted to a subset fields)."""
    """ id can be a single integer or a list of integers. Return trees that exist and 
        the indices of the input id list that were valid."""
    """ Trees returned contain given fields, each fields containing a list of arrays, 
        one for each file chunk. The list corresponds to the valid input ids."""
    """ If no tree is found for the given id, return None"""
    # the tree is all subhalos between SubhaloID and LastProgenitorID
    RowNum, LastProgID, SubhaloID = treeOffsets(basePath, snapNum, id, treeName)

    emptyindex = np.where(np.array(RowNum) == -1)[0]
    if emptyindex.shape[0] > 0:
        print("Warning, empty return. Subhalo", id[emptyindex], "at snapNum", snapNum[emptyindex], "not in tree.")
    
    # remove empty rows
    nonemptyindex = np.where(np.array(RowNum) != -1)[0]
    if nonemptyindex.shape[0] == 0:
        print("Warning, no valid subhalos in tree for id", id)
        return None, nonemptyindex
    if type(RowNum) is not np.ndarray:
        RowNum = [RowNum]
        LastProgID = [LastProgID]
        SubhaloID = [SubhaloID]
    RowNum = np.array(RowNum)[nonemptyindex]
    LastProgID = np.array(LastProgID)[nonemptyindex]
    SubhaloID = np.array(SubhaloID)[nonemptyindex]

    rowStart = RowNum
    rowEnd   = RowNum + (LastProgID - SubhaloID)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    offsets = subLinkOffsets(basePath, treeName, cache)

    # find the tree file chunk containing this row
    rowOffsets = np.tile(np.array(rowStart), (len(offsets), 1)) - np.tile(np.array(offsets), (len(rowStart), 1)).T

    try:
        fileNum = np.empty(rowOffsets.shape[1], dtype=int)
        for i in range(rowOffsets.shape[1]):
            fileNum[i] = np.max(np.where(rowOffsets[:, i] >= 0))
    except ValueError as err:
        print("ERROR: ", err)
        print("rowStart = {}, offsets = {}, rowOffsets = {}".format(rowStart, offsets, rowOffsets))
        print(np.where(rowOffsets >= 0))
        raise
    fileOff = rowOffsets[fileNum, np.arange(len(fileNum))]

    fileNum_New, indices = np.unique(fileNum, return_inverse=True)

    f = []
    for i in range(len(fileNum_New)):
        f.append(h5py.File(treePath(basePath, treeName, fileNum_New[i]), 'r'))

    # load only main progenitor branch? in this case, get MainLeafProgenitorID now
    if onlyMPB:
        MainLeafProgenitorID = [f[indices[i]]['MainLeafProgenitorID'][fileOff[i]] for i in range(len(fileNum))]

        rowEnd = np.array(RowNum) + (np.array(MainLeafProgenitorID) - np.array(SubhaloID))

    # load only main descendant branch (e.g. from z=0 descendant to current subhalo)
    if onlyMDB:
        RootDescendantID = [f[indices[i]]['RootDescendantID'][fileOff[i]] for i in range(len(fileNum))]

        # re-calculate tree subset (rowStart), either single branch to root descendant, or 
        # subset of tree ending at this subhalo if this subhalo is not on the MPB of that 
        # root descendant
        rowStart = np.array(RowNum) - (np.array(SubhaloID) - np.array(RootDescendantID)) + 1
        rowEnd   = np.array(RowNum) + 1
        fileOff -= (rowEnd - rowStart)


    # calculate number of rows to load
    nRows = rowEnd - rowStart + 1

    # read
    result = {'count': nRows}

    # if no fields requested, return all fields
    if not fields:
        fields = list(f[0].keys())
    # read all requested fields
    for field in fields:
        if field not in f[0].keys():
            raise Exception("SubLink tree does not have field ["+field+"]")
        
        # read
        result[field] = [f[indices[i]][field][fileOff[i]:fileOff[i]+nRows[i]] for i in range(len(indices))]
    # close all files
    for i in range(len(f)):
        f[i].close()
    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]

    return result, nonemptyindex



def maxPastMassType (tree, index, partType='stars'):
    """ Get maximum past mass (of the given partType) along the main branch of a subhalo
        specified by index within this tree. """
    ptNum = partTypeNum(partType)

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index] + 1
    masses = tree['SubhaloMassType'][index: index + branchSize, ptNum]
    return np.max(masses)

def numMergersType(tree, minMassRatio=1e-10, massPartType='stars', index=0, alongFullTree=False):
    """ Calculate the number of mergers, along the main progenitor branch, in this sub-tree 
    (optionally above some mass ratio threshold). If alongFullTree, count across the full 
    sub-tree and not only along the MPB. """
    """ mass ratio is defined by a certain partType, e.g. 'stars' or 'gas'."""
    reqfields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMassType']
    
    if not set(reqfields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqfields))
    
    num = 0
    if minMassRatio > 0:
        invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index]
    fpIndex = np.arange(index + 1, index + branchSize + 1)

    mergertime = np.where(tree['NextProgenitorID'][fpIndex] != -1)[0]

    for i in mergertime:
        npID = tree['NextProgenitorID'][fpIndex][i]
        if minMassRatio == 0:
            while npID != -1:
                num += 1
                npID = tree['NextProgenitorID'][index + (npID - rootID)]
                if alongFullTree:
                    if tree['FirstProgenitorID'][index + (npID - rootID)] != -1:
                        numSubtree = numMergers(tree, minMassRatio=minMassRatio, index=index + (npID - rootID), alongFullTree=alongFullTree)
                        num += numSubtree
        else:
            fpMass = maxPastMassType(tree, fpIndex[i], partType=massPartType)
            while npID != -1:
                npIndex = index + (npID - rootID)
                npMass  = maxPastMassType(tree, npIndex, partType=massPartType)

                # count if both masses are non-zero, and ratio exceeds threshold
                if fpMass > 0.0 and npMass > 0.0:
                    ratio = npMass / fpMass

                    if ratio >= minMassRatio and ratio <= invMassRatio:
                        num += 1

                npID = tree['NextProgenitorID'][npIndex]

                # count along full tree instead of just along the MPB? (non-standard)
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        numSubtree = numMergers(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        num += numSubtree
    return num


def maxPastMass(tree, index):
    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index] + 1
    masses = tree['SubhaloMass'][index: index + branchSize]
    return np.max(masses)

def numMergers(tree, minMassRatio=1e-10, index=0, alongFullTree=False):
    """ Calculate the number of mergers, along the main progenitor branch, in this sub-tree 
    (optionally above some mass ratio threshold). If alongFullTree, count across the full 
    sub-tree and not only along the MPB. """
    """ mass ratio is defined by the total subhalo mass, not a specific partType."""
    reqfields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMass']
    
    if not set(reqfields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqfields))
    
    num = 0
    if minMassRatio > 0:
        invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index]
    fpIndex = np.arange(index + 1, index + branchSize + 1)

    mergertime = np.where(tree['NextProgenitorID'][fpIndex] != -1)[0]

    for i in mergertime:
        npID = tree['NextProgenitorID'][fpIndex][i]
        if minMassRatio == 0:
            while npID != -1:
                num += 1
                npID = tree['NextProgenitorID'][index + (npID - rootID)]
                if alongFullTree:
                    if tree['FirstProgenitorID'][index + (npID - rootID)] != -1:
                        numSubtree = numMergers(tree, minMassRatio=minMassRatio, index=index + (npID - rootID), alongFullTree=alongFullTree)
                        num += numSubtree
        else:
            fpMass = maxPastMass(tree, fpIndex[i])
            while npID != -1:
                npIndex = index + (npID - rootID)
                npMass  = maxPastMass(tree, npIndex)

                # count if both masses are non-zero, and ratio exceeds threshold
                if fpMass > 0.0 and npMass > 0.0:
                    ratio = npMass / fpMass

                    if ratio >= minMassRatio and ratio <= invMassRatio:
                        num += 1

                npID = tree['NextProgenitorID'][npIndex]

                # count along full tree instead of just along the MPB? (non-standard)
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        numSubtree = numMergers(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        num += numSubtree
    return num



def extractTree(Tree, index, fields=None):
    """ Extract a single subhalo tree from the list of trees returned by loadTree, specified by index.
        Certain fields can be selected, if not specified, all fields are returned."""
    if not fields:
        fields = list(Tree.keys())

    SingleTree = {}
    for field in fields:
        if field not in Tree.keys():
            raise Exception("SubLink tree does not have field ["+field+"]")
        SingleTree[field] = Tree[field][index]
    return SingleTree



def MergerIDs(tree, index=0, minMassRatio=1e-10, alongFullTree=False):
    """ Calculate the merger tree of a subhalo, specified by index within this tree.
        Returns a list of merger indices, each index is a subhalo ID."""
    reqfields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMass']
    
    if not set(reqfields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqfields))
    
    mergerindex = []
    if minMassRatio > 0:
        invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index]
    fpIndex = np.arange(index + 1, index + branchSize + 1)
    haloIndex = fpIndex - 1

    mergertime = np.where(tree['NextProgenitorID'][fpIndex] != -1)[0]

    for i in mergertime:
        npID = tree['NextProgenitorID'][fpIndex][i]
        if minMassRatio == 0:
            while npID != -1:
                npIndex = index + (npID - rootID)
                mergerindex.append([fpIndex[i], npIndex, haloIndex[i]])
                npID = tree['NextProgenitorID'][npIndex]
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        indexSubtree = MergerIDs(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        mergerindex.extend(indexSubtree)
        else:
            fpMass = maxPastMass(tree, fpIndex[i])
            while npID != -1:
                npIndex = index + (npID - rootID)
                npMass  = maxPastMass(tree, npIndex)

                # count if both masses are non-zero, and ratio exceeds threshold
                if fpMass > 0.0 and npMass > 0.0:
                    ratio = npMass / fpMass

                    if ratio >= minMassRatio and ratio <= invMassRatio:
                        mergerindex.append([fpIndex[i], npIndex, haloIndex[i]])

                npID = tree['NextProgenitorID'][npIndex]

                # count along full tree instead of just along the MPB? (non-standard)
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        indexSubtree = MergerIDs(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        mergerindex.extend(indexSubtree)
    return mergerindex

def Mergers(tree, index=0, minMassRatio=1e-10, alongFullTree=False, fields=None):
    """ Calculate the merger tree of a subhalo, specified by index within this tree.
        Returns a list of merger indices, each index is a subhalo ID."""
    reqfields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMass']
    
    if not set(reqfields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqfields))
    if fields is not None:
        if not set(fields).issubset(tree.keys()):
            raise Exception('Error: Input fields must be a subset of fields in the tree.')
    
    mergerindex = []
    if minMassRatio > 0:
        invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index]
    fpIndex = np.arange(index + 1, index + branchSize + 1)
    haloIndex = fpIndex - 1

    mergertime = np.where(tree['NextProgenitorID'][fpIndex] != -1)[0]

    for i in mergertime:
        npID = tree['NextProgenitorID'][fpIndex][i]
        if minMassRatio == 0:
            while npID != -1:
                npIndex = index + (npID - rootID)
                mergerindex.append([fpIndex[i], npIndex, haloIndex[i]])
                npID = tree['NextProgenitorID'][npIndex]
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        numSubtree = MergerIDs(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        mergerindex.extend(numSubtree)
        else:
            fpMass = maxPastMass(tree, fpIndex[i])
            while npID != -1:
                npIndex = index + (npID - rootID)
                npMass  = maxPastMass(tree, npIndex)

                # count if both masses are non-zero, and ratio exceeds threshold
                if fpMass > 0.0 and npMass > 0.0:
                    ratio = npMass / fpMass

                    if ratio >= minMassRatio and ratio <= invMassRatio:
                        mergerindex.append([fpIndex[i], npIndex, haloIndex[i]])

                npID = tree['NextProgenitorID'][npIndex]

                # count along full tree instead of just along the MPB? (non-standard)
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        numSubtree = MergerIDs(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        mergerindex.extend(numSubtree)
    if fields is None:
        return mergerindex
    else:
        mergers = {}
        mergers['index'] = mergerindex
        for field in fields:
            mergers[field] = tree[field][mergerindex]
        return mergers


def MergerIDsType(tree, index=0, minMassRatio=1e-10, alongFullTree=False, partType='stars'):
    """ Calculate the merger tree of a subhalo, specified by index within this tree.
        Returns a list of merger indices, each index is a subhalo ID."""
    reqfields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMass']
    
    if not set(reqfields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqfields))
    
    mergerindex = []
    if minMassRatio > 0:
        invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index]
    fpIndex = np.arange(index + 1, index + branchSize + 1)
    haloIndex = fpIndex - 1

    mergertime = np.where(tree['NextProgenitorID'][fpIndex] != -1)[0]

    for i in mergertime:
        npID = tree['NextProgenitorID'][fpIndex][i]
        if minMassRatio == 0:
            while npID != -1:
                npIndex = index + (npID - rootID)
                mergerindex.append([fpIndex[i], npIndex, haloIndex[i]])
                npID = tree['NextProgenitorID'][npIndex]
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        indexSubtree = MergerIDsType(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        mergerindex.extend(indexSubtree)
        else:
            fpMass = maxPastMassType(tree, fpIndex[i], partType)
            while npID != -1:
                npIndex = index + (npID - rootID)
                npMass  = maxPastMassType(tree, npIndex, partType)

                # count if both masses are non-zero, and ratio exceeds threshold
                if fpMass > 0.0 and npMass > 0.0:
                    ratio = npMass / fpMass

                    if ratio >= minMassRatio and ratio <= invMassRatio:
                        mergerindex.append([fpIndex[i], npIndex, haloIndex[i]])

                npID = tree['NextProgenitorID'][npIndex]

                # count along full tree instead of just along the MPB? (non-standard)
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        indexSubtree = MergerIDsType(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        mergerindex.extend(indexSubtree)
    return mergerindex

def MergersType(tree, index=0, minMassRatio=1e-10, alongFullTree=False, fields=None, partType='stars'):
    """ Calculate the merger tree of a subhalo, specified by index within this tree.
        Returns a list of merger indices, each index is a subhalo ID."""
    reqfields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID',
                 'FirstProgenitorID', 'SubhaloMass']
    
    if not set(reqfields).issubset(tree.keys()):
        raise Exception('Error: Input tree needs to have loaded fields: '+', '.join(reqfields))
    if fields is not None:
        if not set(fields).issubset(tree.keys()):
            raise Exception('Error: Input fields must be a subset of fields in the tree.')
    
    mergerindex = []
    if minMassRatio > 0:
        invMassRatio = 1.0 / minMassRatio

    # walk back main progenitor branch
    rootID = tree['SubhaloID'][index]

    branchSize = tree['MainLeafProgenitorID'][index] - tree['SubhaloID'][index]
    fpIndex = np.arange(index + 1, index + branchSize + 1)
    haloIndex = fpIndex - 1

    mergertime = np.where(tree['NextProgenitorID'][fpIndex] != -1)[0]

    for i in mergertime:
        npID = tree['NextProgenitorID'][fpIndex][i]
        if minMassRatio == 0:
            while npID != -1:
                npIndex = index + (npID - rootID)
                mergerindex.append([fpIndex[i], npIndex, haloIndex[i]])
                npID = tree['NextProgenitorID'][npIndex]
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        numSubtree = MergerIDsType(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        mergerindex.extend(numSubtree)
        else:
            fpMass = maxPastMassType(tree, fpIndex[i], partType)
            while npID != -1:
                npIndex = index + (npID - rootID)
                npMass  = maxPastMassType(tree, npIndex, partType)

                # count if both masses are non-zero, and ratio exceeds threshold
                if fpMass > 0.0 and npMass > 0.0:
                    ratio = npMass / fpMass

                    if ratio >= minMassRatio and ratio <= invMassRatio:
                        mergerindex.append([fpIndex[i], npIndex, haloIndex[i]])

                npID = tree['NextProgenitorID'][npIndex]

                # count along full tree instead of just along the MPB? (non-standard)
                if alongFullTree:
                    if tree['FirstProgenitorID'][npIndex] != -1:
                        numSubtree = MergerIDsType(tree, minMassRatio=minMassRatio, index=npIndex, alongFullTree=alongFullTree)
                        mergerindex.extend(numSubtree)
    if fields is None:
        return mergerindex
    else:
        mergers = {}
        mergers['index'] = mergerindex
        for field in fields:
            mergers[field] = tree[field][mergerindex]
        return mergers
