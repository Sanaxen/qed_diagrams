from qed_diagrams import (
    Diagram,
    DiagramRequest,
    generate_diagrams,
    is_one_particle_irreducible,
    loop_order_from_counts,
    passes_furry_theorem,
    validate_request,
    vertex_count_from_loops,
)


def test_compton_scattering_is_valid():
    req = DiagramRequest(1, 1, 1, 1, 2)
    valid, _, fermions, photons = validate_request(req)
    assert valid
    assert (fermions, photons) == (1, 0)
    assert generate_diagrams(req) == []


def test_electron_number_is_conserved():
    valid, message, *_ = validate_request(DiagramRequest(1, 0, 2, 1, 2))
    assert not valid
    assert "電荷" in message


def test_odd_photon_stub_count_is_rejected():
    valid, message, *_ = validate_request(DiagramRequest(1, 0, 1, 1, 2))
    assert not valid
    assert "偶数" in message


def test_pair_annihilation_is_valid():
    req = DiagramRequest(1, 0, 0, 2, 2, positron_in=1, loops=0)
    assert validate_request(req)[0]
    assert generate_diagrams(req) == []


def test_one_loop_electron_self_energy_is_valid():
    req = DiagramRequest(1, 0, 1, 0, 2, loops=1)
    assert validate_request(req)[0]
    assert generate_diagrams(req)


def test_wrong_loop_order_is_rejected():
    req = DiagramRequest(1, 1, 1, 1, 2, loops=1)
    valid, message, *_ = validate_request(req)
    assert not valid
    assert "L=0" in message


def test_moller_vertices_have_one_incoming_and_one_outgoing_arrow():
    req = DiagramRequest(2, 0, 2, 0, 2, loops=0)
    diagrams = generate_diagrams(req, limit=10)
    assert diagrams == []


def test_one_loop_self_energy_is_1pi():
    req = DiagramRequest(1, 0, 1, 0, 2, loops=1)
    diagrams = generate_diagrams(req)
    assert diagrams
    assert all(is_one_particle_irreducible(diagram) for diagram in diagrams)


def test_two_loop_vertex_has_seven_nonzero_1pi_diagrams():
    req = DiagramRequest(1, 0, 1, 1, 5, loops=2)
    diagrams = generate_diagrams(req, limit=24, seed=7)
    assert len(diagrams) == 7
    assert all(passes_furry_theorem(diagram) for diagram in diagrams)


def test_three_loop_vertex_has_72_distinct_1pi_diagrams():
    req = DiagramRequest(1, 1, 1, 0, 7, loops=3)
    diagrams = generate_diagrams(req, limit=100)
    assert len(diagrams) == 72
    assert all(passes_furry_theorem(diagram) for diagram in diagrams)


def test_diagram_limit_is_respected():
    req = DiagramRequest(1, 1, 1, 0, 7, loops=3)
    assert len(generate_diagrams(req, limit=12)) == 12


def test_vertex_and_loop_order_are_interchangeable():
    assert vertex_count_from_loops(external_lines=3, loops=2) == 5
    assert loop_order_from_counts(external_lines=3, vertices=5) == 2
    assert loop_order_from_counts(external_lines=3, vertices=4) is None
