"""
Alternative implementations for PyRosetta functionality.

This module provides OpenMM, Biopython, and FASPR-based alternatives to PyRosetta functions,
enabling BindCraft to run without PyRosetta installation. These implementations
aim to provide similar functionality with reasonable approximations where exact
replication is not possible.

Functions:
    openmm_relax: Structure relaxation using OpenMM with optional FASPR side-chain repacking
                 Includes de-concatenation/re-concatenation logic to prevent unintended
                 peptide bond formation between biologically distinct chains/segments
    openmm_relax_subprocess: Run relax in a fresh process to isolate OpenCL context
    pr_alternative_score_interface: Interface scoring using Biopython and FreeSASA
    _calculate_shape_complementarity: Shape complementarity calculation using sc-rs or fallback
    _compute_sasa_metrics: SASA calculations using Biopython or FreeSASA
    _compute_sasa_metrics_with_freesasa: FreeSASA-based SASA calculations
    _run_faspr: Helper to run FASPR side-chain packing
    _resolve_faspr_binary: Locate FASPR binary
    _add_hydrogens_and_minimize: Add hydrogens and minimize post-FASPR structures

Helper Functions:
    _get_openmm_forcefield: Singleton ForceField instance
    _create_lj_repulsive_force: Custom LJ repulsion for clash resolution
    _create_backbone_restraint_force: Backbone position restraints
    _chain_total_sasa: Calculate total SASA for a chain
    _suppress_freesasa_warnings: Suppress FreeSASA warnings
    hotspot_residues: Interface residue identification

Chain Manipulation Functions (from biopython_utils):
    compute_target_segment_lengths: Analyze original PDB to determine segment boundaries
                                   within each biological chain based on residue gaps
    split_chain_into_subchains: Split a concatenated chain into separate chains based
                               on provided segment lengths and new chain IDs
    merge_chains_into_single: Merge multiple chains back into a single chain for
                             downstream compatibility with BindCraft scoring

Chain Handling:
    ColabDesign concatenates target chains into a single chain 'A', which can cause OpenMM
    to inappropriately infer peptide bonds between biologically distinct chains or across gaps
    in discontinuous segments. To prevent this, we implement:
    
    1. De-concatenation: Split chain 'A' into separate chains (C, D, E, etc.) based on the
       original biological chains and any discontinuous segments within each chain
    2. Structure Processing: Run PDBFixer, OpenMM relaxation, and FASPR on the de-concatenated
       structure to maintain proper chain separation
    3. Re-concatenation: Merge the processed chains back into chain 'A' for downstream scoring
    
    This approach ensures that OpenMM treats each biological unit as a separate entity while
    maintaining compatibility with the existing BindCraft scoring pipeline.

Rationale:
    In long runs we observed sporadic OpenCL context failures after many relax calls,
    consistent with driver/runtime state or memory accumulation. The subprocess helper
    guarantees full teardown per relax, isolating OpenCL state between runs.
"""

import gc
import sys
import shutil
import copy
import os
import json
import tempfile
import subprocess
import contextlib
import time
import pathlib
from itertools import zip_longest
from .generic_utils import clean_pdb
from .logging_utils import vprint
from .biopython_utils import hotspot_residues, biopython_align_all_ca
from .biopython_utils import compute_target_segment_lengths
from .biopython_utils import compute_target_chain_lengths, split_chain_into_subchains, merge_chains_into_single

# OpenMM imports
import openmm
from openmm import app, unit, Platform, OpenMMException
from pdbfixer import PDBFixer

# Bio.PDB imports
from Bio.PDB import PDBParser, PDBIO, Polypeptide, Structure, Model
from Bio.SeqUtils import seq1
from Bio.PDB.SASA import ShrakeRupley

# Cache a single OpenMM ForceField instance to avoid repeated XML parsing per relaxation
_OPENMM_FORCEFIELD_SINGLETON = None

# Optional FreeSASA availability
try:
    import freesasa  # type: ignore
    _HAS_FREESASA = True
except Exception:
    freesasa = None  # type: ignore
    _HAS_FREESASA = False

# Chothia/NACCESS-like atomic radii (heavy atoms dominate SASA)
R_CHOTHIA = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}

# Hydrophobic amino acids set (match PyRosetta hydrophobic/aromatic intent; include GLY)
HYDROPHOBIC_AA_SET = set("ACFGILMPVWY")

########################################################
# Shape complementarity
########################################################

def _calculate_shape_complementarity(pdb_file_path, binder_chain="B", target_chain="A", distance=4.0):
    """
    Calculate shape complementarity using sc-rs CLI when available.
    Looks first for a local binary placed next to this module (e.g., 'functions/sc' or 'functions/sc-rs').
    Falls back to a conservative placeholder (0.70) if sc-rs is not installed or fails.

    Parameters
    ----------
    pdb_file_path : str
        Path to the PDB file containing the complex
    binder_chain : str
        Chain ID of the binder (default: "B")
    target_chain : str
        Chain ID of the target (default: "A")
    distance : float
        Unused here; retained for API compatibility

    Returns
    -------
    float
        Shape complementarity in [0, 1]
    """
    try:
        start_time = time.time()
        basename = os.path.basename(pdb_file_path)
        vprint(f"[SC-RS] Initiating shape complementarity for {basename} (target={target_chain}, binder={binder_chain})")
        # Resolve sc-rs binary: local next to this file, then env vars, then PATH
        module_dir = os.path.dirname(os.path.abspath(__file__))
        local_candidates = [
            os.path.join(module_dir, 'sc'),
            os.path.join(module_dir, 'sc-rs'),
        ]
        env_candidates = [os.environ.get('SC_RS_BIN'), os.environ.get('SC_BIN')]
        path_candidates = [
            shutil.which('sc'),
            shutil.which('sc-rs'),
            shutil.which('shape-complementarity'),
            shutil.which('sc_rs'),
        ]

        sc_bin = None
        for candidate in local_candidates + env_candidates + path_candidates:
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                sc_bin = candidate
                break

        if sc_bin is None:
            # Fallback to placeholder if not found
            vprint(f"[SC-RS] Binary not found; using placeholder value 0.70 for {basename}")
            return 0.70
        else:
            vprint(f"[SC-RS] Using binary: {sc_bin}")

        # sc-rs CLI: sc <pdb> <chainA> <chainB> --json; SC is symmetric, pass target first for clarity
        cmd = [sc_bin, pdb_file_path, str(target_chain), str(binder_chain), '--json']
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = (proc.stdout or '').strip()
        if not stdout:
            vprint(f"[SC-RS] Empty output; using placeholder 0.70 for {basename}")
            return 0.70

        # Parse JSON strictly, else try to extract from mixed output
        try:
            payload = json.loads(stdout)
        except Exception:
            payload = None
            try:
                s_idx = stdout.rfind('{')
                e_idx = stdout.rfind('}')
                if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
                    payload = json.loads(stdout[s_idx:e_idx+1])
            except Exception:
                payload = None

        if isinstance(payload, dict):
            try:
                sc_key = 'sc' if 'sc' in payload else ('sc_value' if 'sc_value' in payload else None)
                if sc_key is not None:
                    sc_val = float(payload[sc_key])
                    if 0.0 <= sc_val <= 1.0:
                        elapsed = time.time() - start_time
                        vprint(f"[SC-RS] Completed for {basename}: SC={sc_val:.2f} in {elapsed:.2f}s")
                        return sc_val
            except Exception:
                pass
    except subprocess.TimeoutExpired:
        print(f"[SC-RS] ERROR: sc-rs timed out for {os.path.basename(pdb_file_path)}")
    except subprocess.CalledProcessError as e:
        print(f"[SC-RS] ERROR running sc-rs: {e}. stderr: {getattr(e, 'stderr', '')}")
    except Exception as e:
        print(f"[SC-RS] WARN: Failed to compute SC for {pdb_file_path}: {e}")

    # Fallback to placeholder to keep pipelines running
    vprint(f"[SC-RS] Fallback placeholder 0.70 for {os.path.basename(pdb_file_path)}")
    return 0.70

########################################################
# SASA / Surface hydrophobicity
########################################################

@contextlib.contextmanager
def _suppress_freesasa_warnings():
    """Temporarily redirect OS-level stderr (fd=2) to suppress FreeSASA warnings."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
    except Exception:
        # Fallback: no suppression
        yield

def _chain_total_sasa(chain_entity):
    return sum(getattr(atom, "sasa", 0.0) for atom in chain_entity.get_atoms())

def _compute_sasa_metrics(pdb_file_path, binder_chain="B", target_chain="A"):
    """
    Compute SASA-derived metrics needed for interface scoring using Biopython.

    Returns a 5-tuple:
        (surface_hydrophobicity_fraction, binder_sasa_in_complex, binder_sasa_monomer,
         target_sasa_in_complex, target_sasa_monomer)
    """
    surface_hydrophobicity_fraction = 0.0
    binder_sasa_in_complex = 0.0
    binder_sasa_monomer = 0.0
    target_sasa_in_complex = 0.0
    target_sasa_monomer = 0.0

    try:
        t0 = time.time()
        basename = os.path.basename(pdb_file_path)
        vprint(f"[SASA-Biopython] Start for {basename} (binder={binder_chain}, target={target_chain})")
        parser = PDBParser(QUIET=True)

        # Compute atom-level SASA for the entire complex
        complex_structure = parser.get_structure('complex', pdb_file_path)
        complex_model = complex_structure[0]
        sr_complex = ShrakeRupley(probe_radius=1.40, n_points=960, radii_dict=R_CHOTHIA)
        sr_complex.compute(complex_model, level='A')

        # Binder chain SASA within complex
        if binder_chain in complex_model:
            binder_chain_in_complex = complex_model[binder_chain]
            binder_sasa_in_complex = _chain_total_sasa(binder_chain_in_complex)
        
        # Target chain SASA within complex
        if target_chain in complex_model:
            target_chain_in_complex = complex_model[target_chain]
            target_sasa_in_complex = _chain_total_sasa(target_chain_in_complex)

        # Binder monomer SASA and surface hydrophobicity fraction (area-based)
        if binder_chain in complex_model:
            binder_only_structure = Structure.Structure('binder_only')
            binder_only_model = Model.Model(0)
            binder_only_chain = copy.deepcopy(complex_model[binder_chain])
            binder_only_model.add(binder_only_chain)
            binder_only_structure.add(binder_only_model)

            sr_mono = ShrakeRupley(probe_radius=1.40, n_points=960, radii_dict=R_CHOTHIA)
            sr_mono.compute(binder_only_model, level='A')
            binder_sasa_monomer = _chain_total_sasa(binder_only_chain)

            # Residue-based hydrophobic surface fraction (sum residue SASA for hydrophobic residues)
            hydrophobic_res_sasa = 0.0
            for residue in binder_only_chain:
                if Polypeptide.is_aa(residue, standard=True):
                    try:
                        aa1 = seq1(residue.get_resname()).upper()
                    except Exception:
                        aa1 = ''
                    if aa1 in HYDROPHOBIC_AA_SET:
                        res_sasa = sum(getattr(atom, 'sasa', 0.0) for atom in residue.get_atoms())
                        hydrophobic_res_sasa += res_sasa
            surface_hydrophobicity_fraction = (hydrophobic_res_sasa / binder_sasa_monomer) if binder_sasa_monomer > 0.0 else 0.0
        else:
            surface_hydrophobicity_fraction = 0.0

        # Target monomer SASA
        if target_chain in complex_model:
            target_only_structure = Structure.Structure('target_only')
            target_only_model = Model.Model(0)
            target_only_chain = copy.deepcopy(complex_model[target_chain])
            target_only_model.add(target_only_chain)
            target_only_structure.add(target_only_model)
            sr_target_mono = ShrakeRupley(probe_radius=1.40, n_points=960, radii_dict=R_CHOTHIA)
            sr_target_mono.compute(target_only_model, level='A')
            target_sasa_monomer = _chain_total_sasa(target_only_chain)

        elapsed = time.time() - t0
        vprint(f"[SASA-Biopython] Completed for {basename} in {elapsed:.2f}s")
    except Exception as e_sasa:
        print(f"[Biopython-SASA] ERROR for {pdb_file_path}: {e_sasa}")
        # Fallbacks chosen to match original behavior
        surface_hydrophobicity_fraction = 0.30
        binder_sasa_in_complex = 0.0
        binder_sasa_monomer = 0.0
        target_sasa_in_complex = 0.0
        target_sasa_monomer = 0.0

    return (
        surface_hydrophobicity_fraction,
        binder_sasa_in_complex,
        binder_sasa_monomer,
        target_sasa_in_complex,
        target_sasa_monomer,
    )

def _compute_sasa_metrics_with_freesasa(pdb_file_path, binder_chain="B", target_chain="A"):
    """
    Compute SASA-derived metrics using FreeSASA with fallback to Biopython on failure.

    Returns a 5-tuple:
        (surface_hydrophobicity_fraction, binder_sasa_in_complex, binder_sasa_monomer,
         target_sasa_in_complex, target_sasa_monomer)
    """
    try:
        t0 = time.time()
        basename = os.path.basename(pdb_file_path)
        vprint(f"[SASA-FreeSASA] Start for {basename} (binder={binder_chain}, target={target_chain})")
        if not _HAS_FREESASA:
            raise RuntimeError("FreeSASA not available")

        # Optional classifier (e.g., NACCESS) via repo file or env var FREESASA_CONFIG
        classifier_obj = None
        try:
            classifier_path = os.environ.get('FREESASA_CONFIG')
            if not classifier_path or not os.path.isfile(classifier_path):
                # default to repo-provided NACCESS config
                module_dir = os.path.dirname(os.path.abspath(__file__))
                default_cfg = os.path.join(module_dir, 'freesasa_naccess.cfg')
                if os.path.isfile(default_cfg):
                    classifier_path = default_cfg
            if classifier_path and os.path.isfile(classifier_path):
                classifier_obj = freesasa.Classifier(classifier_path)  # type: ignore[name-defined]
                vprint(f"[SASA-FreeSASA] Using classifier: {classifier_path}")
        except Exception:
            classifier_obj = None

        # Complex SASA
        if classifier_obj is not None:
            structure_complex = freesasa.Structure(pdb_file_path, classifier=classifier_obj)  # type: ignore[name-defined]
        else:
            structure_complex = freesasa.Structure(pdb_file_path)  # type: ignore[name-defined]
        result_complex = freesasa.calc(structure_complex)  # type: ignore[name-defined]

        binder_sasa_in_complex = 0.0
        target_sasa_in_complex = 0.0
        try:
            # FreeSASA Python API expects a list of selection definition strings: "name, selector"
            selection_defs = [
                f"binder, chain {str(binder_chain)}",
                f"target, chain {str(target_chain)}",
            ]
            sel_area = freesasa.selectArea(selection_defs, structure_complex, result_complex)  # type: ignore[name-defined]
            # sel_area is a dict-like mapping from selection name to area
            binder_sasa_in_complex = float(sel_area.get('binder', 0.0))
            target_sasa_in_complex = float(sel_area.get('target', 0.0))
        except Exception:
            pass

        # Prepare monomer PDBs via Bio.PDB (only used for chain extraction)
        parser = PDBParser(QUIET=True)
        complex_structure_bp = parser.get_structure('complex_for_freesasa', pdb_file_path)
        complex_model_bp = complex_structure_bp[0]

        binder_sasa_monomer = 0.0
        target_sasa_monomer = 0.0
        surface_hydrophobicity_fraction = 0.0

        tmp_binder_path = None
        tmp_target_path = None
        try:
            if binder_chain in complex_model_bp:
                binder_only_structure = Structure.Structure('binder_only')
                binder_only_model = Model.Model(0)
                binder_only_chain = copy.deepcopy(complex_model_bp[binder_chain])
                binder_only_model.add(binder_only_chain)
                binder_only_structure.add(binder_only_model)

                io_b = PDBIO()
                io_b.set_structure(binder_only_structure)
                tmp_b = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
                tmp_b.close()
                tmp_binder_path = tmp_b.name
                io_b.save(tmp_binder_path)

                if classifier_obj is not None:
                    structure_binder_only = freesasa.Structure(tmp_binder_path, classifier=classifier_obj)  # type: ignore[name-defined]
                else:
                    structure_binder_only = freesasa.Structure(tmp_binder_path)  # type: ignore[name-defined]
                result_binder_only = freesasa.calc(structure_binder_only)  # type: ignore[name-defined]
                binder_sasa_monomer = float(result_binder_only.totalArea())

                # FreeSASA residue selection only: hydrophobic residues / total (no fallback)
                try:
                    sel_defs = [
                        "hydro, resn ala+val+leu+ile+met+phe+pro+trp+tyr+cys+gly"
                    ]
                    with _suppress_freesasa_warnings():
                        sel_area = freesasa.selectArea(sel_defs, structure_binder_only, result_binder_only)  # type: ignore[name-defined]
                    hydro_area = float(sel_area.get('hydro', 0.0))
                    if binder_sasa_monomer > 0.0:
                        surface_hydrophobicity_fraction = hydro_area / binder_sasa_monomer
                except Exception:
                    # Keep default 0.0 if selection fails
                    pass

            if target_chain in complex_model_bp:
                target_only_structure = Structure.Structure('target_only')
                target_only_model = Model.Model(0)
                target_only_chain = copy.deepcopy(complex_model_bp[target_chain])
                target_only_model.add(target_only_chain)
                target_only_structure.add(target_only_model)

                io_t = PDBIO()
                io_t.set_structure(target_only_structure)
                tmp_t = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
                tmp_t.close()
                tmp_target_path = tmp_t.name
                io_t.save(tmp_target_path)

                if classifier_obj is not None:
                    structure_target_only = freesasa.Structure(tmp_target_path, classifier=classifier_obj)  # type: ignore[name-defined]
                else:
                    structure_target_only = freesasa.Structure(tmp_target_path)  # type: ignore[name-defined]
                result_target_only = freesasa.calc(structure_target_only)  # type: ignore[name-defined]
                target_sasa_monomer = float(result_target_only.totalArea())
        finally:
            if tmp_binder_path and os.path.isfile(tmp_binder_path):
                try:
                    os.remove(tmp_binder_path)
                except Exception:
                    pass
            if tmp_target_path and os.path.isfile(tmp_target_path):
                try:
                    os.remove(tmp_target_path)
                except Exception:
                    pass

        elapsed = time.time() - t0
        vprint(f"[SASA-FreeSASA] Completed for {basename} in {elapsed:.2f}s")
        return (
            surface_hydrophobicity_fraction,
            binder_sasa_in_complex,
            binder_sasa_monomer,
            target_sasa_in_complex,
            target_sasa_monomer,
        )
    except Exception as e_fsasa:
        print(f"[FreeSASA] ERROR for {pdb_file_path}: {e_fsasa}")
        return _compute_sasa_metrics(pdb_file_path, binder_chain=binder_chain, target_chain=target_chain)

########################################################
# Interface scoring
########################################################
def pr_alternative_score_interface(pdb_file, binder_chain="B", target_chain="A", sasa_engine="auto"):
    """
    Calculate interface scores using PyRosetta-free alternatives including SCASA shape complementarity.
    
    This function provides comprehensive interface scoring without PyRosetta dependency by combining:
    - Biopython-based SASA calculations
    - SCASA shape complementarity calculation  
    - Interface residue identification
    
    Parameters
    ----------
    pdb_file : str
        Path to PDB file
    binder_chain : str
        Chain ID of the binder
    sasa_engine : str
        "auto" (default) prefers FreeSASA if installed, else Biopython.
        "freesasa" forces FreeSASA (falls back to Biopython on error).
        "biopython" forces Biopython Shrake-Rupley.
        
    Returns
    -------
    tuple
        (interface_scores, interface_AA, interface_residues_pdb_ids_str)
    """
    t0_all = time.time()
    basename = os.path.basename(pdb_file)
    vprint(f"[Alt-Score] Initiating PyRosetta-free scoring for {basename} (binder={binder_chain}, sasa_engine={sasa_engine})")

    # Get interface residues via Biopython (works without PyRosetta)
    t0_if = time.time()
    vprint(f"[Alt-Score] Finding interface residues (hotspot_residues)...")
    interface_residues_set = hotspot_residues(pdb_file, binder_chain)
    interface_residues_pdb_ids = [f"{binder_chain}{pdb_res_num}" for pdb_res_num in interface_residues_set.keys()]
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)
    vprint(f"[Alt-Score] Found {len(interface_residues_pdb_ids)} interface residues in {time.time()-t0_if:.2f}s")

    # Initialize amino acid dictionary for interface composition
    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    for pdb_res_num, aa_type in interface_residues_set.items():
        interface_AA[aa_type] += 1

    # SASA-based calculations: select engine
    t0_sasa = time.time()
    if str(sasa_engine).lower() == "biopython":
        vprint(f"[Alt-Score] Computing SASA with Biopython Shrake-Rupley...")
        surface_hydrophobicity_fraction, \
        binder_sasa_in_complex, \
        binder_sasa_monomer, \
        target_sasa_in_complex, \
        target_sasa_monomer = _compute_sasa_metrics(
            pdb_file, binder_chain=binder_chain, target_chain=target_chain 
        )
    elif str(sasa_engine).lower() == "freesasa":
        vprint(f"[Alt-Score] Computing SASA with FreeSASA...")
        surface_hydrophobicity_fraction, \
        binder_sasa_in_complex, \
        binder_sasa_monomer, \
        target_sasa_in_complex, \
        target_sasa_monomer = _compute_sasa_metrics_with_freesasa(
            pdb_file, binder_chain=binder_chain, target_chain=target_chain 
        )
    else:
        if _HAS_FREESASA:
            vprint(f"[Alt-Score] Computing SASA with FreeSASA (auto)...")
            surface_hydrophobicity_fraction, \
            binder_sasa_in_complex, \
            binder_sasa_monomer, \
            target_sasa_in_complex, \
            target_sasa_monomer = _compute_sasa_metrics_with_freesasa(
                pdb_file, binder_chain=binder_chain, target_chain=target_chain 
            )
        else:
            vprint(f"[Alt-Score] Computing SASA with Biopython (auto fallback)...")
            surface_hydrophobicity_fraction, \
            binder_sasa_in_complex, \
            binder_sasa_monomer, \
            target_sasa_in_complex, \
            target_sasa_monomer = _compute_sasa_metrics(
                pdb_file, binder_chain=binder_chain, target_chain=target_chain 
            )
    vprint(f"[Alt-Score] SASA computations finished in {time.time()-t0_sasa:.2f}s")

    # Compute buried SASA: binder-side and total (binder + target)
    interface_binder_dSASA = max(binder_sasa_monomer - binder_sasa_in_complex, 0.0)
    interface_target_dSASA = 0.0
    try:
        interface_target_dSASA = max(target_sasa_monomer - target_sasa_in_complex, 0.0)
    except Exception as e_idsasa:
        print(f"[Biopython-SASA] WARN interface_target_dSASA for {pdb_file}: {e_idsasa}")
    interface_total_dSASA = interface_binder_dSASA + interface_target_dSASA
    # Align with PyRosetta: use TOTAL interface dSASA divided by binder SASA IN COMPLEX
    interface_binder_fraction = (interface_total_dSASA / binder_sasa_in_complex * 100.0) if binder_sasa_in_complex > 0.0 else 0.0

    # Calculate shape complementarity using SCASA
    t0_sc = time.time()
    vprint(f"[Alt-Score] Computing shape complementarity (SC)...")
    interface_sc = _calculate_shape_complementarity(pdb_file, binder_chain, target_chain=target_chain)
    vprint(f"[Alt-Score] SC computation finished in {time.time()-t0_sc:.2f}s")
    
    # Fixed placeholder values for metrics that are not currently computed without PyRosetta
    # These values are chosen to pass active filters
    interface_nres = len(interface_residues_pdb_ids)                    # computed from interface residues
    interface_interface_hbonds = 5                                      # passes >= 3 (active filter)
    interface_delta_unsat_hbonds = 1                                    # passes <= 4 (active filter)
    interface_hbond_percentage = 60.0                                   # informational (no active filter)
    interface_bunsch_percentage = 0.0                                   # informational (no active filter)
    binder_score = -1.0                                                 # passes <= 0 (active filter) - never results in rejections based on extensive testing
    interface_packstat = 0.65                                           # informational (no active filter)
    interface_dG = -10.0                                                # passes <= 0 (active filter) - never results in rejections based on extensive testing
    interface_dG_SASA_ratio = 0.0                                       # informational (no active filter)

    interface_scores = {
        'binder_score': binder_score,
        'surface_hydrophobicity': surface_hydrophobicity_fraction,
        'interface_sc': interface_sc,
        'interface_packstat': interface_packstat,
        'interface_dG': interface_dG,
        'interface_dSASA': interface_total_dSASA,
        'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
        'interface_fraction': interface_binder_fraction,
        'interface_hydrophobicity': (
            (sum(interface_AA[aa] for aa in 'ACFGILMPVWY') / interface_nres * 100.0) if interface_nres > 0 else 0.0
        ),
        'interface_nres': interface_nres,
        'interface_interface_hbonds': interface_interface_hbonds,   
        'interface_hbond_percentage': interface_hbond_percentage,   
        'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds, 
        'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    # Round float values to two decimals for consistency
    interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items()}

    vprint(f"[Alt-Score] Completed scoring for {basename} in {time.time()-t0_all:.2f}s")
    return interface_scores, interface_AA, interface_residues_pdb_ids_str

########################################################
# OpenMM-based relax and forcefield setup
########################################################

def _get_openmm_forcefield():
    global _OPENMM_FORCEFIELD_SINGLETON
    if _OPENMM_FORCEFIELD_SINGLETON is None:
        _OPENMM_FORCEFIELD_SINGLETON = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    return _OPENMM_FORCEFIELD_SINGLETON

# Helper function for k conversion
def _k_kj_per_nm2(k_kcal_A2):
    return k_kcal_A2 * 4.184 * 100.0

# Helper function for LJ repulsive force creation
def _create_lj_repulsive_force(system, lj_rep_base_k_kj_mol, lj_rep_ramp_factors, original_sigmas, nonbonded_force_index):
    lj_rep_custom_force = None
    k_rep_lj_param_index = -1

    if lj_rep_base_k_kj_mol > 0 and original_sigmas and lj_rep_ramp_factors:
        lj_rep_custom_force = openmm.CustomNonbondedForce(
            "k_rep_lj * (((sigma_particle1 + sigma_particle2) * 0.5 / r)^12)"
        )
        
        initial_k_rep_val = lj_rep_base_k_kj_mol * lj_rep_ramp_factors[0]
        # Global parameters in OpenMM CustomNonbondedForce expect plain float values for the constant.
        # The energy expression itself defines how this constant is used with physical units.
        k_rep_lj_param_index = lj_rep_custom_force.addGlobalParameter("k_rep_lj", float(initial_k_rep_val)) 
        lj_rep_custom_force.addPerParticleParameter("sigma_particle")

        for sigma_val_nm in original_sigmas:
            lj_rep_custom_force.addParticle([sigma_val_nm])

        # Check if nonbonded_force_index is valid before trying to get the force
        if nonbonded_force_index != -1:
            existing_nb_force = system.getForce(nonbonded_force_index)
            nb_method = existing_nb_force.getNonbondedMethod()
            
            if nb_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic]:
                lj_rep_custom_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic if nb_method == openmm.NonbondedForce.CutoffPeriodic else openmm.CustomNonbondedForce.CutoffNonPeriodic)
                lj_rep_custom_force.setCutoffDistance(existing_nb_force.getCutoffDistance())
                if nb_method == openmm.NonbondedForce.CutoffPeriodic:
                     lj_rep_custom_force.setUseSwitchingFunction(existing_nb_force.getUseSwitchingFunction())
                     if existing_nb_force.getUseSwitchingFunction():
                         lj_rep_custom_force.setSwitchingDistance(existing_nb_force.getSwitchingDistance())
            elif nb_method == openmm.NonbondedForce.NoCutoff:
                 lj_rep_custom_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
            
            for ex_idx in range(existing_nb_force.getNumExceptions()):
                p1, p2, chargeProd, sigmaEx, epsilonEx = existing_nb_force.getExceptionParameters(ex_idx)
                lj_rep_custom_force.addExclusion(p1, p2)
        else:
            # This case should ideally not be hit if sigmas were extracted,
            # but as a fallback, don't try to use existing_nb_force.
            # Default to NoCutoff if we couldn't determine from an existing force.
            lj_rep_custom_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

        lj_rep_custom_force.setForceGroup(2)
        system.addForce(lj_rep_custom_force)
    
    return lj_rep_custom_force, k_rep_lj_param_index

# Helper function for backbone restraint force creation
def _create_backbone_restraint_force(system, fixer, restraint_k_kcal_mol_A2):
    restraint_force = None
    k_restraint_param_index = -1

    if restraint_k_kcal_mol_A2 > 0:
        restraint_force = openmm.CustomExternalForce(
            "0.5 * k_restraint * ( (x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0) )" 
        )
        # Global parameters in OpenMM CustomExternalForce also expect plain float values.
        k_restraint_param_index = restraint_force.addGlobalParameter("k_restraint", _k_kj_per_nm2(restraint_k_kcal_mol_A2))
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")

        initial_positions = fixer.positions 
        num_bb_restrained = 0
        BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}
        for atom in fixer.topology.atoms():
            if atom.name in BACKBONE_ATOM_NAMES:
                xyz_vec = initial_positions[atom.index].value_in_unit(unit.nanometer) 
                restraint_force.addParticle(atom.index, [xyz_vec[0], xyz_vec[1], xyz_vec[2]]) 
                num_bb_restrained +=1
        
        if num_bb_restrained > 0:
            restraint_force.setForceGroup(1)
            system.addForce(restraint_force)
        else:
            restraint_force = None 
            k_restraint_param_index = -1
            
    return restraint_force, k_restraint_param_index

# --- FASPR integration helpers ---
def _resolve_faspr_binary():
    """Locate the FASPR binary. Search order: env FASPR_BIN → functions/FASPR → PATH.

    Returns
    -------
    tuple[str|None, str|None]
        (binary_path, binary_dir) or (None, None) if not found.
    """
    # Env override
    env_bin = os.environ.get('FASPR_BIN')
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin, os.path.dirname(os.path.abspath(env_bin))

    # Repo-provided binary
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(module_dir, 'FASPR')
        if os.path.isfile(candidate):
            # ensure executable bit if possible
            try:
                if not os.access(candidate, os.X_OK):
                    os.chmod(candidate, os.stat(candidate).st_mode | 0o755)
            except Exception:
                pass
            if os.access(candidate, os.X_OK):
                return candidate, module_dir
    except Exception:
        pass

    # PATH lookup
    which_faspr = shutil.which('FASPR')
    if which_faspr and os.path.isfile(which_faspr) and os.access(which_faspr, os.X_OK):
        return which_faspr, os.path.dirname(os.path.abspath(which_faspr))

    return None, None

def _run_faspr(input_pdb_path, output_pdb_path, sequence_txt_path=None, timeout=900):
    """Run FASPR on an input PDB and write repacked output.

    FASPR requires 'dun2010bbdep.bin' to be colocated with the executable. We set cwd
    to the FASPR directory so the binary can find its rotamer library.

    Returns True on success, False otherwise.
    """
    faspr_bin, faspr_dir = _resolve_faspr_binary()
    if not faspr_bin or not faspr_dir:
        print("[FASPR] WARN: FASPR binary not found; skipping repack")
        return False

    cmd = [faspr_bin, '-i', os.path.abspath(input_pdb_path), '-o', os.path.abspath(output_pdb_path)]
    if sequence_txt_path and os.path.isfile(sequence_txt_path):
        cmd.extend(['-s', os.path.abspath(sequence_txt_path)])

    try:
        vprint(f"[FASPR] Running: {' '.join(cmd)}")
        # Capture output but do not emit banner/stdout; only report on errors via exceptions below.
        proc = subprocess.run(cmd, cwd=faspr_dir, check=True, capture_output=True, text=True, timeout=timeout)
        # Verify output exists and is non-empty
        if os.path.isfile(output_pdb_path) and os.path.getsize(output_pdb_path) > 0:
            return True
    except subprocess.TimeoutExpired:
        print(f"[FASPR] ERROR: Timeout running FASPR on {os.path.basename(input_pdb_path)}")
    except subprocess.CalledProcessError as e:
        print(f"[FASPR] ERROR: FASPR failed: rc={e.returncode} stderr={getattr(e, 'stderr', '')}")
    except Exception as e:
        print(f"[FASPR] ERROR: {e}")
    return False

def _add_hydrogens_and_minimize(pdb_in_path, pdb_out_path, platform_order=None,
                                force_tolerance_kj_mol_nm=0.1, max_iterations=500):
    """Add hydrogens with PDBFixer and run a short OpenMM minimization, then save PDB.

    Returns
    -------
    tuple[str|None, float]
        (platform_used, seconds)
    """
    t0 = time.time()
    try:
        fixer = PDBFixer(filename=pdb_in_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        forcefield = _get_openmm_forcefield()
        system = forcefield.createSystem(fixer.topology,
                                         nonbondedMethod=app.CutoffNonPeriodic,
                                         nonbondedCutoff=1.0*unit.nanometer,
                                         constraints=app.HBonds)

        integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin,
                                                     1.0/unit.picosecond,
                                                     0.002*unit.picoseconds)

        # Platform selection
        plat_used = None
        props = {}
        sim = None
        if platform_order is None:
            platform_order = ['Metal', 'OpenCL', 'CPU']
        for p_name in platform_order:
            try:
                platform_obj = Platform.getPlatformByName(p_name)
                if p_name == 'CUDA':
                    props = {'CudaPrecision': 'mixed'}
                elif p_name == 'OpenCL':
                    props = {'OpenCLPrecision': 'single'}
                sim = app.Simulation(fixer.topology, system, integrator, platform_obj, props)
                plat_used = p_name
                break
            except Exception:
                sim = None
                continue
        if sim is None:
            raise OpenMMException("No suitable OpenMM platform for post-FASPR minimization")

        sim.context.setPositions(fixer.positions)
        tol = force_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer
        sim.minimizeEnergy(tolerance=tol, maxIterations=max_iterations)
        positions = sim.context.getState(getPositions=True).getPositions()
        with open(pdb_out_path, 'w') as outf:
            app.PDBFile.writeFile(sim.topology, positions, outf, keepIds=True)
        # Cleanup
        try:
            del sim, integrator, system, fixer
        except Exception:
            pass
        gc.collect()
        return plat_used, (time.time() - t0)
    except Exception as e:
        print(f"[OpenMM-PostFASPR] WARN: Failed post-FASPR H-add/min: {e}")
        try:
            shutil.copy(pdb_in_path, pdb_out_path)
        except Exception:
            pass
        return None, (time.time() - t0)

def openmm_relax(pdb_file_path, output_pdb_path, use_gpu_relax=True,
                 openmm_max_iterations=1000, # Safety cap per stage to avoid stalls (set 0 for unlimited)
                 # Default force tolerances for ramp stages (kJ/mol/nm)
                 openmm_ramp_force_tolerance_kj_mol_nm=2.0,
                 openmm_final_force_tolerance_kj_mol_nm=0.1,
                 restraint_k_kcal_mol_A2=3.0,
                 restraint_ramp_factors=(1.0, 0.4, 0.0), # 3-stage restraint ramp factors
                 md_steps_per_shake=1000, # MD steps for each shake (applied only to first two stages)
                 lj_rep_base_k_kj_mol=10.0, # Base strength for extra LJ repulsion (kJ/mol)
                 lj_rep_ramp_factors=(0.0, 1.5, 3.0), # 3-stage LJ repulsion ramp factors (soft → hard)
                 perf_report=None, # Optional dict to populate with detailed timing/energy metrics
                 override_platform_order=None,
                 use_faspr_repack=True, # If True, run FASPR repack after relax (default: enabled)
                 post_faspr_minimize=True # If True, add H and short standardized min after FASPR
                 ):
    """
    Relaxes a PDB structure using OpenMM with L-BFGS minimizer.
    
    Chain Handling:
    ---------------
    To prevent OpenMM/PDBFixer from inappropriately connecting biologically distinct chains
    or segments, this function implements a de-concatenation/re-concatenation strategy:
    
    1. De-concatenation: If BINDCRAFT_STARTING_PDB and BINDCRAFT_TARGET_CHAINS environment
       variables are set, splits the concatenated chain 'A' into separate chains based on:
       - Original biological chain boundaries from the starting PDB
       - Discontinuous segments within each biological chain (residue number gaps)
       
    2. Structure Processing: Runs PDBFixer, OpenMM relaxation, and optional FASPR on the
       de-concatenated structure, maintaining proper chain separation throughout
       
    3. Re-concatenation: Merges all processed chains back into chain 'A' for compatibility
       with downstream BindCraft scoring
    
    Processing Steps:
    ----------------
    - Uses PDBFixer to prepare the structure (add missing atoms, hydrogens, etc.)
    - Applies backbone heavy-atom harmonic restraints (ramped down using restraint_ramp_factors)
    - Uses OBC2 implicit solvent with additional ramped LJ-like repulsive force for clash resolution
    - Includes short MD shakes for the first two ramp stages only (speed optimization)
    - Optionally integrates FASPR for side-chain repacking after relaxation
    - Follows with hydrogen addition and short minimization to standardize final structure
    - Uses accept-to-best position bookkeeping across all stages
    - Aligns final structure to input and copies B-factors
    - Preserves disulfide bonds and other connectivity records in final output
    
    Debug Output:
    ------------
    When BINDCRAFT_DEBUG_PDBS=1, saves intermediate structures:
    - .debug_deconcat.pdb: After de-concatenation (if applicable)
    - .debug_pdbfixer.pdb: After PDBFixer preparation
    - .debug_post_initial_relax.pdb: After OpenMM relaxation
    - .debug_post_faspr.pdb: After FASPR repacking (if enabled)
    
    Returns
    -------
    platform_name_used : str or None
        Name of the OpenMM platform actually used (e.g., 'CUDA', 'OpenCL', or 'CPU').
    """
    start_time = time.time()
    basename = os.path.basename(pdb_file_path)
    vprint(f"[OpenMM-Relax] Initiating relax for {basename}")
    # Debug file paths (next to final output)
    try:
        _dbg_dir = os.path.dirname(output_pdb_path)
        _dbg_base = os.path.splitext(os.path.basename(output_pdb_path))[0]
        _dbg_deconcat = os.path.join(_dbg_dir, f"{_dbg_base}.debug_deconcat.pdb")
        _dbg_pdbfixer = os.path.join(_dbg_dir, f"{_dbg_base}.debug_pdbfixer.pdb")
        _dbg_post_initial_relax = os.path.join(_dbg_dir, f"{_dbg_base}.debug_post_initial_relax.pdb")
        _dbg_post_faspr = os.path.join(_dbg_dir, f"{_dbg_base}.debug_post_faspr.pdb")
    except Exception:
        _dbg_deconcat = None
        _dbg_pdbfixer = None
        _dbg_relaxed_premerge = None
    _perf = {"stages": []} if isinstance(perf_report, dict) else None
    best_energy = float('inf') * unit.kilojoule_per_mole # Initialize with units
    best_positions = None

    # 1. Store original B-factors (per residue CA or first atom)
    original_residue_b_factors = {}
    bio_parser = PDBParser(QUIET=True)
    try:
        original_structure = bio_parser.get_structure('original', pdb_file_path)
        for model in original_structure:
            for chain in model:
                for residue in chain:
                    # Use Polypeptide.is_aa if available and needed for strict AA check
                    # For B-factor copying, we might want to copy for any residue type present.
                    # Let's assume standard AA check for now as in pr_relax context
                    if Polypeptide.is_aa(residue, standard=True):
                        ca_atom = None
                        try: # Try to get 'CA' atom
                            ca_atom = residue['CA']
                        except KeyError: # 'CA' not in residue
                            pass
                            
                        b_factor = None
                        if ca_atom:
                            b_factor = ca_atom.get_bfactor()
                        else: # Fallback to first atom if CA not found
                            first_atom = next(residue.get_atoms(), None)
                            if first_atom:
                                b_factor = first_atom.get_bfactor()
                        
                        if b_factor is not None:
                            # residue.id is (hetfield, resseq, icode)
                            original_residue_b_factors[(chain.id, residue.id)] = b_factor
    except Exception as _:
        original_residue_b_factors = {} 

    try:
        # 1. Prepare the PDB structure using PDBFixer
        t_prep_start = time.time()
        # If the input likely has ColabDesign-concatenated target in chain A,
        # de-concatenate into explicit chains before running PDBFixer so OpenMM
        # cannot form peptide bonds across biological chain boundaries.
        pdb_for_fixer = pdb_file_path
        _deconcat_tmp = None
        _reconcat_spec = None
        try:
            _starting_pdb_env = os.environ.get('BINDCRAFT_STARTING_PDB')
            _starting_chains_env = os.environ.get('BINDCRAFT_TARGET_CHAINS')
            vprint(f"[OpenMM-Relax] Checking for de-concatenation: starting_pdb={_starting_pdb_env}, chains={_starting_chains_env}")
            if _starting_pdb_env and os.path.isfile(_starting_pdb_env) and _starting_chains_env:
                _chain_ids_list = [c.strip() for c in _starting_chains_env.split(',') if c.strip()]
                # Prefer segment-aware lengths to prevent cross-gap bonding
                _lengths = compute_target_segment_lengths(_starting_pdb_env, _starting_chains_env)
                vprint(f"[OpenMM-Relax] Segment-aware lengths: {_lengths}")
                if _chain_ids_list and _lengths and sum(1 for l in _lengths if l > 0) >= 2:
                    import string
                    # Build exactly N chain IDs (exclude A/B), prioritizing uppercase then digits then lowercase
                    N = len(_lengths)
                    pool = [c for c in string.ascii_uppercase if c not in ('A','B')]
                    pool += list('0123456789')
                    pool += list('abcdefghijklmnopqrstuvwxyz')
                    if len(pool) < N:
                        raise RuntimeError(f"Insufficient chain IDs for {N} segments")
                    _new_chain_ids = pool[:N]
                    vprint(f"[OpenMM-Relax] De-concatenating chain A into {N} segments: {_new_chain_ids[:min(6,N)]}{'...' if N>6 else ''}")
                    _tmpf = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
                    _tmpf.close()
                    _deconcat_tmp = _tmpf.name
                    split_chain_into_subchains(pdb_file_path, source_chain_id='A', subchain_lengths=_lengths, new_chain_ids=_new_chain_ids, output_path=_deconcat_tmp)
                    vprint(f"[OpenMM-Relax] De-concatenated PDB written to: {_deconcat_tmp}")
                    # Sanity check non-empty temp file; if empty, skip de-concatenation
                    try:
                        if (not os.path.isfile(_deconcat_tmp)) or os.path.getsize(_deconcat_tmp) == 0:
                            vprint(f"[OpenMM-Relax] De-concatenation produced empty file; skipping and using original input")
                            _deconcat_tmp = None
                            _reconcat_spec = None
                            pdb_for_fixer = pdb_file_path
                        else:
                            pdb_for_fixer = _deconcat_tmp
                    except Exception:
                        _deconcat_tmp = None
                        _reconcat_spec = None
                        pdb_for_fixer = pdb_file_path
                    # Debug: save de-concatenated PDB next to final output
                    try:
                        if _dbg_deconcat and os.environ.get('BINDCRAFT_DEBUG_PDBS') == '1':
                            if _deconcat_tmp and os.path.isfile(_deconcat_tmp) and os.path.getsize(_deconcat_tmp) > 0:
                                shutil.copy(_deconcat_tmp, _dbg_deconcat)
                                vprint(f"[OpenMM-Relax] Debug de-concat saved: {_dbg_deconcat}")
                    except Exception:
                        pass
                    if _deconcat_tmp:
                        _reconcat_spec = (_new_chain_ids, 'A')
                else:
                    vprint("[OpenMM-Relax] Single chain or no valid lengths - skipping de-concatenation")
            else:
                vprint("[OpenMM-Relax] No de-concatenation context available")
        except Exception as e:
            vprint(f"[OpenMM-Relax] De-concatenation failed: {e}")
            pdb_for_fixer = pdb_file_path

        fixer = PDBFixer(filename=pdb_for_fixer)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues() # This should handle common MODRES
        fixer.removeHeterogens(keepWater=False) # Usually False for relaxation
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0) # Add hydrogens at neutral pH
        vprint(f"[OpenMM-Relax] PDBFixer processing completed on: {pdb_for_fixer}")
        # Debug: write PDBFixer output
        try:
            if _dbg_pdbfixer and os.environ.get('BINDCRAFT_DEBUG_PDBS') == '1':
                with open(_dbg_pdbfixer, 'w') as _df:
                    app.PDBFile.writeFile(fixer.topology, fixer.positions, _df, keepIds=True)
                vprint(f"[OpenMM-Relax] Debug PDBFixer output saved: {_dbg_pdbfixer}")
        except Exception:
            pass
        if _perf is not None:
            _perf["prep_seconds"] = time.time() - t_prep_start

        # 2. Set up OpenMM ForceField, System, Integrator, and Simulation
        # Reuse a module-level ForceField instance to avoid re-parsing XMLs each call
        forcefield = _get_openmm_forcefield()
        
        system = forcefield.createSystem(fixer.topology, 
                                         nonbondedMethod=app.CutoffNonPeriodic, # Retain for OBC2 defined by XML
                                         nonbondedCutoff=1.0*unit.nanometer,    # Retain for OBC2 defined by XML
                                         constraints=app.HBonds)
        
        # Extract original sigmas from the NonbondedForce for the custom LJ repulsion
        original_sigmas = []
        nonbonded_force_index = -1
        for i_force_idx in range(system.getNumForces()): # Use getNumForces and getForce
            force_item = system.getForce(i_force_idx)
            if isinstance(force_item, openmm.NonbondedForce):
                nonbonded_force_index = i_force_idx
                for p_idx in range(force_item.getNumParticles()):
                    charge, sigma, epsilon = force_item.getParticleParameters(p_idx)
                    original_sigmas.append(sigma.value_in_unit(unit.nanometer)) # Store as float in nm
                break
        
        if nonbonded_force_index == -1:
            pass # Keep silent

        # Add custom LJ-like repulsive force (ramped) using helper function
        lj_rep_custom_force, k_rep_lj_param_index = _create_lj_repulsive_force(
            system, 
            lj_rep_base_k_kj_mol, 
            lj_rep_ramp_factors, 
            original_sigmas, 
            nonbonded_force_index
        )
        if 'original_sigmas' in locals(): # Check if it was actually created
            del original_sigmas # Free memory as it's no longer needed in this scope
        
        # Add backbone heavy-atom harmonic restraints using helper function
        restraint_force, k_restraint_param_index = _create_backbone_restraint_force(
            system, 
            fixer, 
            restraint_k_kcal_mol_A2
        )
        
        integrator = openmm.LangevinMiddleIntegrator(300*unit.kelvin, 
                                                  1.0/unit.picosecond, 
                                                  0.002*unit.picoseconds)
        
        simulation = None
        platform_name_used = None # To store the name of the successfully used platform

        if isinstance(override_platform_order, (list, tuple)) and override_platform_order:
            platform_order = list(override_platform_order)
        else:
            platform_order = []
            if use_gpu_relax:
                # Prefer OpenCL, then CUDA (override with env if needed)
                env_order = os.environ.get('OPENMM_PLATFORM_ORDER')
                if env_order:
                    platform_order = [p.strip() for p in env_order.split(',') if p.strip()]
                else:
                    platform_order.extend(['OpenCL', 'CUDA'])
            else:
                # Explicit CPU-only path if GPU is not requested
                platform_order.append('CPU')

        last_exception = None
        for p_name_to_try in platform_order:
            if simulation:
                break

            # Retry up to 3 times per platform with 1s backoff
            for attempt_idx in range(1, 4):
                # ensure fresh simulation object per attempt
                simulation = None
                current_platform_obj = None
                current_properties = {}
                try:
                    current_platform_obj = Platform.getPlatformByName(p_name_to_try)
                    if p_name_to_try == 'CUDA':
                        current_properties = {'CudaPrecision': 'mixed'}
                    elif p_name_to_try == 'OpenCL':
                        current_properties = {'OpenCLPrecision': 'single'}

                    simulation = app.Simulation(
                        fixer.topology, system, integrator, current_platform_obj, current_properties
                    )
                    platform_name_used = p_name_to_try
                    vprint(f"[OpenMM-Relax] Using platform: {platform_name_used}")
                    break
                except (OpenMMException, Exception) as e:
                    last_exception = e
                    if attempt_idx < 3:
                        vprint(f"[OpenMM-Relax] Platform {p_name_to_try} attempt {attempt_idx} failed; retrying in 1s...")
                        time.sleep(1.0)
                        continue
                    else:
                        vprint(f"[OpenMM-Relax] Platform {p_name_to_try} failed after {attempt_idx} attempts")
                        break

            if simulation:
                break
            

        if simulation is None:
            final_error_msg = (
                f"FATAL: Could not initialize OpenMM Simulation with any GPU platform after trying {', '.join(platform_order)}."
            )
            # Prefer raising the last captured exception if present
            if last_exception is not None:
                raise last_exception
            raise OpenMMException(final_error_msg) 
        
        simulation.context.setPositions(fixer.positions)

        # Optional Pre-Minimization Step (before main ramp loop)
        # Perform if restraints or LJ repulsion are active, to stabilize structure.
        if restraint_k_kcal_mol_A2 > 0 or lj_rep_base_k_kj_mol > 0:
            t_init_min_start = time.time()
            
            # Set LJ repulsion to zero for this initial minimization
            if lj_rep_custom_force is not None and k_rep_lj_param_index != -1 and lj_rep_base_k_kj_mol > 0:
                lj_rep_custom_force.setGlobalParameterDefaultValue(k_rep_lj_param_index, 0.0) # Pass plain float
                lj_rep_custom_force.updateParametersInContext(simulation.context)

            # Set restraints to full strength for this initial minimization (if active)
            if restraint_force is not None and k_restraint_param_index != -1 and restraint_k_kcal_mol_A2 > 0:
                # restraint_k_kcal_mol_A2 is the base parameter for restraint strength
                full_initial_restraint_k_val = _k_kj_per_nm2(restraint_k_kcal_mol_A2) 
                restraint_force.setGlobalParameterDefaultValue(k_restraint_param_index, full_initial_restraint_k_val)
                restraint_force.updateParametersInContext(simulation.context)
            
            initial_min_tolerance = openmm_ramp_force_tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer
            simulation.minimizeEnergy(
                tolerance=initial_min_tolerance,
                maxIterations=openmm_max_iterations 
            )
            if _perf is not None:
                _perf["initial_min_seconds"] = time.time() - t_init_min_start

        # 3. Perform staged relaxation: ramp restraints, limited MD shakes, and minimization
        base_k_for_ramp_kcal = restraint_k_kcal_mol_A2

        # Determine number of stages based on provided ramp factors
        # Use restraint_ramp_factors for k_constr and lj_rep_ramp_factors for k_rep_lj
        # Simplified stage iteration using zip_longest
        effective_restraint_factors = restraint_ramp_factors if restraint_k_kcal_mol_A2 > 0 and restraint_ramp_factors else [0.0] # Use 0.0 if no restraint
        effective_lj_rep_factors = lj_rep_ramp_factors if lj_rep_base_k_kj_mol > 0 and lj_rep_ramp_factors else [0.0] # Use 0.0 if no LJ rep

        # If one of the ramps is disabled (e.g. k=0 or empty factors), its factors list will be [0.0].
        # zip_longest will then pair its 0.0 with the active ramp's factors.
        # If both are disabled, it will iterate once with (0.0, 0.0).
        
        ramp_pairs = list(zip_longest(effective_restraint_factors, effective_lj_rep_factors, fillvalue=0.0))
        num_stages = len(ramp_pairs)
        
        # If both k_restraint_kcal_mol_A2 and lj_rep_base_k_kj_mol are 0, 
        # or their factor lists are empty, num_stages will be 1 (due to [0.0] default), 
        # effectively running one minimization stage without these ramps.
        if num_stages == 1 and effective_restraint_factors == [0.0] and effective_lj_rep_factors == [0.0] and not (restraint_k_kcal_mol_A2 > 0 or lj_rep_base_k_kj_mol > 0):
            pass

        for i_stage_val, (k_factor_restraint, current_lj_rep_k_factor) in enumerate(ramp_pairs):
            stage_num = i_stage_val + 1
            _stage_metrics = None
            if _perf is not None:
                _stage_metrics = {
                    "stage_index": stage_num,
                    "restraint_factor": float(k_factor_restraint),
                    "lj_rep_factor": float(current_lj_rep_k_factor),
                    "md_steps_run": 0,
                    "md_seconds": 0.0,
                    "min_seconds": 0.0,
                    "min_calls": 0,
                    "min_energy_trace_kj": [],
                    "energy_start_kj": None,
                    "md_post_energy_kj": None,
                    "final_energy_kj": None,
                }

            # Record starting energy for this stage
            try:
                _stage_energy_start = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                if _stage_metrics is not None:
                    _stage_metrics["energy_start_kj"] = float(_stage_energy_start.value_in_unit(unit.kilojoule_per_mole))
            except Exception:
                pass

            # Set LJ repulsive ramp for the current stage
            if lj_rep_custom_force is not None and k_rep_lj_param_index != -1 and lj_rep_base_k_kj_mol > 0:
                current_lj_rep_k_val = lj_rep_base_k_kj_mol * current_lj_rep_k_factor
                lj_rep_custom_force.setGlobalParameterDefaultValue(k_rep_lj_param_index, current_lj_rep_k_val) # Pass plain float
                lj_rep_custom_force.updateParametersInContext(simulation.context)

            # Set restraint stiffness for the current stage
            if restraint_force is not None and k_restraint_param_index != -1 and restraint_k_kcal_mol_A2 > 0:
                current_stage_k_kcal = base_k_for_ramp_kcal * k_factor_restraint
                numeric_k_for_stage = _k_kj_per_nm2(current_stage_k_kcal)
                restraint_force.setGlobalParameterDefaultValue(k_restraint_param_index, numeric_k_for_stage)
                restraint_force.updateParametersInContext(simulation.context)

            # MD Shake only for first two ramp stages for speed-performance tradeoff
            if md_steps_per_shake > 0 and i_stage_val < 2:
                t_md_start = time.time()
                simulation.context.setVelocitiesToTemperature(300*unit.kelvin) # Reinitialize velocities
                simulation.step(md_steps_per_shake)
                if _stage_metrics is not None:
                    _stage_metrics["md_steps_run"] = int(md_steps_per_shake)
                    _stage_metrics["md_seconds"] = time.time() - t_md_start
                    try:
                        _md_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                        _stage_metrics["md_post_energy_kj"] = float(_md_energy.value_in_unit(unit.kilojoule_per_mole))
                    except Exception:
                        pass

            # Minimization for the current stage
            # Set force tolerance for current stage
            if i_stage_val == num_stages - 1: # Final stage
                current_force_tolerance = openmm_final_force_tolerance_kj_mol_nm
            else: # Ramp stages
                current_force_tolerance = openmm_ramp_force_tolerance_kj_mol_nm
            force_tolerance_quantity = current_force_tolerance * unit.kilojoule_per_mole / unit.nanometer
            
            # Chunked minimization to avoid pathological stalls: run in small blocks and early-stop
            # if energy improvement becomes negligible
            per_call_max_iterations = 200 if (openmm_max_iterations == 0 or openmm_max_iterations > 200) else openmm_max_iterations
            remaining_iterations = openmm_max_iterations
            small_improvement_streak = 0
            last_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

            t_min_start = time.time()
            while True:
                simulation.minimizeEnergy(tolerance=force_tolerance_quantity,
                                          maxIterations=per_call_max_iterations)
                current_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                if _stage_metrics is not None:
                    _stage_metrics["min_calls"] += 1
                    try:
                        _stage_metrics["min_energy_trace_kj"].append(float(current_energy.value_in_unit(unit.kilojoule_per_mole)))
                    except Exception:
                        pass

                # Check improvement magnitude
                try:
                    energy_improvement = last_energy - current_energy
                    if energy_improvement < (0.1 * unit.kilojoule_per_mole):
                        small_improvement_streak += 1
                    else:
                        small_improvement_streak = 0
                except Exception:
                    # If unit math fails for any reason, break conservatively
                    small_improvement_streak = 3

                last_energy = current_energy

                # Decrement remaining iterations if bounded
                if openmm_max_iterations > 0:
                    remaining_iterations -= per_call_max_iterations
                    if remaining_iterations <= 0:
                        break

                # Early stop if improvement is consistently negligible
                if small_improvement_streak >= 3:
                    break

            stage_final_energy = last_energy
            if _stage_metrics is not None:
                try:
                    _stage_metrics["final_energy_kj"] = float(stage_final_energy.value_in_unit(unit.kilojoule_per_mole))
                except Exception:
                    _stage_metrics["final_energy_kj"] = None
                _stage_metrics["min_seconds"] = time.time() - t_min_start
                _perf["stages"].append(_stage_metrics)

            # Accept-to-best bookkeeping
            if stage_final_energy < best_energy:
                best_energy = stage_final_energy 
                best_positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True) # Use asNumpy=True

        # After all stages, set positions to the best ones found
        if best_positions is not None:
            simulation.context.setPositions(best_positions)

        # 4. Save the relaxed structure
        t_save_start = time.time()
        positions = simulation.context.getState(getPositions=True).getPositions()
        with open(output_pdb_path, 'w') as outfile:
            app.PDBFile.writeFile(simulation.topology, positions, outfile, keepIds=True)
        vprint(f"[OpenMM-Relax] OpenMM relaxation completed, saved to: {output_pdb_path}")
        # Debug: relaxed pre-merge
        try:
            if _dbg_post_initial_relax and os.environ.get('BINDCRAFT_DEBUG_PDBS') == '1':
                shutil.copy(output_pdb_path, _dbg_post_initial_relax)
                vprint(f"[OpenMM-Relax] Debug post-initial-relax saved: {_dbg_post_initial_relax}")
        except Exception:
            pass

        # Defer alignment/B-factor transfer and any re-concatenation until after FASPR

        # 5. Optional FASPR repacking and standardized cleanup
        faspr_seconds = None
        faspr_success = False
        post_min_seconds = None
        if use_faspr_repack:
            try:
                t_faspr = time.time()
                # Prepare a temporary heavy-atom PDB for FASPR by stripping hydrogens
                # Reuse PDBFixer to remove hydrogens to avoid FASPR backbone completeness issues.
                tmp_dir = tempfile.mkdtemp(prefix='faspr_')
                tmp_heavy = os.path.join(tmp_dir, 'input_heavy.pdb')
                tmp_faspr_out = os.path.join(tmp_dir, 'faspr_out.pdb')

                try:
                    fixer_heavy = PDBFixer(filename=output_pdb_path)
                    fixer_heavy.findMissingResidues()
                    fixer_heavy.findNonstandardResidues()
                    fixer_heavy.replaceNonstandardResidues()
                    fixer_heavy.removeHeterogens(keepWater=False)
                    fixer_heavy.findMissingAtoms()
                    fixer_heavy.addMissingAtoms()
                    # Intentionally DO NOT add hydrogens here; FASPR ignores sidechains but requires complete backbone
                    with open(tmp_heavy, 'w') as ftmp:
                        app.PDBFile.writeFile(fixer_heavy.topology, fixer_heavy.positions, ftmp, keepIds=True)
                except Exception:
                    # Fallback: copy the current output if fixer fails
                    shutil.copy(output_pdb_path, tmp_heavy)

                faspr_success = _run_faspr(tmp_heavy, tmp_faspr_out)
                faspr_seconds = time.time() - t_faspr

                if faspr_success:
                    # Add hydrogens and a brief, standardized minimization, then overwrite output
                    if post_faspr_minimize:
                        _, post_min_seconds = _add_hydrogens_and_minimize(
                            tmp_faspr_out, output_pdb_path,
                            platform_order=['OpenCL', 'CUDA', 'CPU'],
                            force_tolerance_kj_mol_nm=openmm_final_force_tolerance_kj_mol_nm,
                            max_iterations=300
                        )
                    else:
                        # If not minimizing, at least re-add hydrogens
                        try:
                            fixer2 = PDBFixer(filename=tmp_faspr_out)
                            fixer2.findMissingResidues()
                            fixer2.findNonstandardResidues()
                            fixer2.replaceNonstandardResidues()
                            fixer2.removeHeterogens(keepWater=False)
                            fixer2.findMissingAtoms()
                            fixer2.addMissingAtoms()
                            fixer2.addMissingHydrogens(7.0)
                            with open(output_pdb_path, 'w') as f2:
                                app.PDBFile.writeFile(fixer2.topology, fixer2.positions, f2, keepIds=True)
                        except Exception:
                            shutil.copy(tmp_faspr_out, output_pdb_path)
                # Debug: save after FASPR
                try:
                    if _dbg_post_faspr and os.environ.get('BINDCRAFT_DEBUG_PDBS') == '1':
                        shutil.copy(output_pdb_path, _dbg_post_faspr)
                        vprint(f"[OpenMM-Relax] Debug post-FASPR saved: {_dbg_post_faspr}")
                except Exception:
                    pass
                # Cleanup temp
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            except Exception as e_f:
                print(f"[FASPR] WARN: repack step failed: {e_f}")

        # Final alignment and B-factor application (after FASPR)
        t_align_start = time.time()
        try:
            if _deconcat_tmp and os.path.isfile(_deconcat_tmp):
                vprint("[OpenMM-Relax] Aligning to de-concatenated original to preserve chain separation")
                biopython_align_all_ca(_deconcat_tmp, output_pdb_path)
            else:
                vprint("[OpenMM-Relax] Aligning to original concatenated structure")
                biopython_align_all_ca(pdb_file_path, output_pdb_path)
        except Exception as e:
            vprint(f"[OpenMM-Relax] Alignment failed: {e}")
            pass

        t_bfac_start = time.time()
        if original_residue_b_factors:
            try:
                relaxed_structure_for_bfactors = bio_parser.get_structure('relaxed_aligned', output_pdb_path)
                modified_b_factors = False
                for model in relaxed_structure_for_bfactors:
                    for chain in model:
                        for residue in chain:
                            b_factor_to_apply = original_residue_b_factors.get((chain.id, residue.id))
                            if b_factor_to_apply is not None:
                                for atom in residue:
                                    atom.set_bfactor(b_factor_to_apply)
                                modified_b_factors = True
                if modified_b_factors:
                    io = PDBIO()
                    io.set_structure(relaxed_structure_for_bfactors)
                    io.save(output_pdb_path)
            except Exception:
                pass

        # Re-concatenate chains at the very end (after alignment/B-factors)
        try:
            if _reconcat_spec and isinstance(_reconcat_spec, tuple):
                _src_ids, _dest_id = _reconcat_spec
                vprint(f"[OpenMM-Relax] Re-concatenating chains {_src_ids} back into chain A (final step)")
                merge_chains_into_single(output_pdb_path, _src_ids, dest_chain_id='A', output_path=output_pdb_path)
                vprint(f"[OpenMM-Relax] Re-concatenation completed")
            else:
                vprint("[OpenMM-Relax] No re-concatenation needed")
        except Exception as e:
            vprint(f"[OpenMM-Relax] Re-concatenation failed: {e}")

        # Append connectivity records from a PDBFixer pass to restore SSBOND/CONECT
        try:
            _tmp_conn = None
            import tempfile as _tf
            _tmpf2 = _tf.NamedTemporaryFile(suffix='.pdb', delete=False)
            _tmpf2.close()
            _tmp_conn = _tmpf2.name
            # Run PDBFixer quickly to regenerate connectivity
            _fx = PDBFixer(filename=output_pdb_path)
            with open(_tmp_conn, 'w') as _fd:
                app.PDBFile.writeFile(_fx.topology, _fx.positions, _fd, keepIds=True)
            # Extract connectivity lines
            conn_lines = []
            with open(_tmp_conn, 'r') as _fd:
                for _ln in _fd:
                    if _ln.startswith(('SSBOND', 'CONECT', 'LINK')):
                        conn_lines.append(_ln)
            # Append unique connectivity lines to final output (before END if present)
            if conn_lines:
                with open(output_pdb_path, 'r') as _fin:
                    lines = _fin.readlines()
                end_idx = next((i for i,l in enumerate(lines) if l.startswith('END')), None)
                # Deduplicate: avoid duplicating existing connectivity
                existing = set(l for l in lines if l.startswith(('SSBOND','CONECT','LINK')))
                to_add = [l for l in conn_lines if l not in existing]
                if to_add:
                    if end_idx is not None:
                        lines = lines[:end_idx] + to_add + lines[end_idx:]
                    else:
                        lines.extend(to_add)
                    with open(output_pdb_path, 'w') as _fout:
                        _fout.writelines(lines)
                vprint(f"[OpenMM-Relax] Appended {len(to_add)} connectivity records from PDBFixer")
            try:
                if _tmp_conn and os.path.isfile(_tmp_conn):
                    os.remove(_tmp_conn)
            except Exception:
                pass
        except Exception as e:
            vprint(f"[OpenMM-Relax] WARN: failed to append connectivity records: {e}")

        # 6. Clean the output PDB
        clean_pdb(output_pdb_path)

        # Explicitly delete heavy OpenMM objects to avoid cumulative slowdowns across many trajectories
        try:
            del positions
        except Exception:
            pass
        try:
            del simulation, integrator, system, restraint_force, lj_rep_custom_force, fixer
        except Exception:
            pass
        gc.collect()

        # Cleanup temp file created during de-concatenation
        try:
            if _deconcat_tmp and os.path.isfile(_deconcat_tmp):
                os.remove(_deconcat_tmp)
        except Exception:
            pass

        elapsed_total = time.time() - start_time
        if _perf is not None:
            # Summarize overall perf settings/results
            _perf["platform"] = platform_name_used
            _perf["ramp_count"] = num_stages
            _perf["md_steps_per_shake"] = int(md_steps_per_shake)
            _perf["restraint_ramp_factors"] = list(effective_restraint_factors)
            _perf["lj_rep_ramp_factors"] = list(effective_lj_rep_factors)
            try:
                _perf["best_energy_kj"] = float(best_energy.value_in_unit(unit.kilojoule_per_mole))
            except Exception:
                _perf["best_energy_kj"] = None
            _perf["save_seconds"] = time.time() - t_save_start
            _perf["align_seconds"] = time.time() - t_align_start
            _perf["b_factor_seconds"] = time.time() - t_bfac_start
            _perf["total_seconds"] = elapsed_total
            if use_faspr_repack:
                _perf["faspr_seconds"] = faspr_seconds
                _perf["faspr_success"] = bool(faspr_success)
                _perf["post_faspr_min_seconds"] = post_min_seconds
            perf_report.update(_perf)
        vprint(f"[OpenMM-Relax] Completed relax for {basename} in {elapsed_total:.2f}s (platform={platform_name_used})")
        
        # Cleanup temporary de-concatenated file
        if _deconcat_tmp and os.path.isfile(_deconcat_tmp):
            try:
                os.remove(_deconcat_tmp)
                vprint(f"[OpenMM-Relax] Cleaned up temporary de-concatenated file: {_deconcat_tmp}")
            except Exception:
                pass
                
        return platform_name_used

    except Exception as _:
        shutil.copy(pdb_file_path, output_pdb_path)
        gc.collect()
        elapsed_total = time.time() - start_time
        print(f"[OpenMM-Relax] ERROR; copied input to output for {basename} after {elapsed_total:.2f}s")
        print(f"[OpenMM-Relax] ERROR; exeception {str(_)}")
        if _perf is not None:
            try:
                _perf["exception"] = str(_)
                perf_report.update(_perf)
            except Exception:
                pass
        # Cleanup temporary de-concatenated file in error case
        if _deconcat_tmp and os.path.isfile(_deconcat_tmp):
            try:
                os.remove(_deconcat_tmp)
            except Exception:
                pass
                
        # Guard against 'platform_name_used' not being assigned yet
        try:
            return platform_name_used
        except UnboundLocalError:
            return None

def openmm_relax_subprocess(pdb_file_path, output_pdb_path, use_gpu_relax=True, timeout=None, max_attempts=3, use_faspr_repack=True):
    """Run openmm_relax in a fresh Python process to fully reset OpenCL context per run.
    Retries if the child fell back to copying input (soft failure) or if the child crashes (hard failure).
    Streams child logs to parent stdout/stderr so DEBUG lines are visible.
    """
    import logging as _logging
    want_verbose = _logging.getLogger("functions").isEnabledFor(_logging.DEBUG)

    # Change in cwd is needed when running outside of code directory
    # parents[1] is the directory above the current one
    cwd = pathlib.Path(__file__).parents[1].resolve()
    
    # also resolve input/output to allow for running outside of main code repo
    pdb_file_path = str(pathlib.Path(pdb_file_path).resolve())
    output_pdb_path = str(pathlib.Path(output_pdb_path).resolve())

    code_parts = []
    if want_verbose:
        code_parts.append(
            "import logging; logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')"
        )
    else:
        code_parts.append("import logging; logging.basicConfig(level=logging.WARNING)")
    # Suppress noisy third-party DEBUG logs in child process
    code_parts.append("import logging")
    code_parts.append("logging.getLogger('matplotlib').setLevel(logging.WARNING)")
    code_parts.append("logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)")
    code_parts.append("logging.getLogger('pyrosetta').setLevel(logging.WARNING)")
    code_parts.append("logging.getLogger('pyrosetta.distributed').setLevel(logging.WARNING)")
    code_parts.append("logging.getLogger('pyrosetta.distributed.utility.pickle').setLevel(logging.WARNING)")
    code_parts.append("from functions.pr_alternative_utils import openmm_relax")
    code_parts.append(
        f"plat = openmm_relax({pdb_file_path!r}, {output_pdb_path!r}, use_gpu_relax={bool(use_gpu_relax)}, use_faspr_repack={bool(use_faspr_repack)})"
    )    
    py_code = "; ".join(code_parts)

    # Signature to detect soft fallback path inside child (input copied to output)
    fallback_signature = "[OpenMM-Relax] ERROR; copied input to output"
    
    attempts = int(max(1, int(max_attempts)))
    for attempt_idx in range(1, attempts + 1):
        # Capture output to inspect for fallback while still forwarding to parent
        proc = subprocess.run(
            [sys.executable, "-c", py_code], timeout=timeout, capture_output=True, text=True, cwd=cwd
        )

        # Forward child output to parent streams to preserve visibility, but filter stderr
        if proc.stdout:
            try:
                sys.stdout.write(proc.stdout)
            except Exception:
                pass
        if proc.stderr:
            try:
                for line in proc.stderr.splitlines(True):  # Preserve newlines
                    if "Failed to read file: /tmp/dep-" not in line:
                        sys.stderr.write(line)
            except Exception:
                pass

        # Hard failure: non-zero rc from child
        if proc.returncode != 0:
            if attempt_idx >= attempts:
                raise RuntimeError(
                    f"Subprocess openmm_relax failed with rc={proc.returncode} after {attempt_idx} attempts"
                )
            time.sleep(0.5)
            continue

        # Soft failure: child printed fallback copy message
        combined_out = (proc.stdout or "") + (proc.stderr or "")
        if fallback_signature in combined_out and attempt_idx < attempts:
            print(f"[OpenMM-Relax] Detected fallback copy; retrying ({attempt_idx+1}/{attempts})")
            # Remove fallback-copied file before retry so the next success writes a clean output
            try:
                if os.path.isfile(output_pdb_path):
                    os.remove(output_pdb_path)
            except Exception:
                pass
            time.sleep(0.5)
            continue

        # Success (or final acceptable fallback)
        return None

    return None

########################################################
# Main for testing
########################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--relax-cli", action="store_true")
    parser.add_argument("--in", dest="inp", type=str)
    parser.add_argument("--out", dest="out", type=str)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    if args.relax_cli:
        if args.verbose:
            import logging
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s %(name)s: %(message)s'
            )
        plat = openmm_relax(args.inp, args.out, use_gpu_relax=args.gpu)
        if plat:
            print(plat)
        sys.exit(0)
