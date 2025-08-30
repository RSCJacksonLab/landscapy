import types

import fitness_landscape.core.superscape as S


def test_from_parallel_construction_gpu_and_parent_cpu(monkeypatch):
    # Capture options passed to Ray task
    captured = {"num_gpus": [], "num_cpus": []}

    class FakeTask:
        def options(self, num_gpus=0, num_cpus=1.0):
            captured["num_gpus"].append(num_gpus)
            captured["num_cpus"].append(num_cpus)
            return self

        def remote(self, **job):
            # return an opaque reference object
            return object()

    # Monkeypatch the Ray remote task and ray.wait/get
    monkeypatch.setattr(S, "_create_landscape_task", FakeTask())

    class DummyRef:
        pass

    # ray.wait returns the list of refs one by one
    def fake_wait(pending, num_returns=1, timeout=30.0):
        if not pending:
            return [], set()
        ref = next(iter(pending))
        rest = set(pending)
        rest.discard(ref)
        return [ref], rest

    def fake_get(ref):
        # Return a trivial FitnessLandscape
        from fitness_landscape.core.sequence import generate_sequences
        from fitness_landscape.core.fitness import NumericFitness
        from fitness_landscape.core.landscape import FitnessLandscape
        seqs = generate_sequences(length=1, alphabet=[0, 1])
        layers = {"default": NumericFitness(name="default", values=[[1.0] for _ in seqs])}
        return FitnessLandscape.from_sequences(seqs, fitness_layers=layers)

    import ray
    monkeypatch.setattr(ray, "wait", fake_wait)
    monkeypatch.setattr(ray, "get", lambda ref: fake_get(ref))
    monkeypatch.setattr(ray, "is_initialized", lambda: True)

    jobs = [{
        "sequences": [],
        "graph_type": "evol_diffusion",
        "embedding_domain": "plm",
        "_compute_embeddings": True,
    }]

    ss = S.FitnessSuperscape.from_parallel_construction(
        constructor_type='undirected',
        construction_jobs=jobs,
        _parent_task_cpus=0.25,
        burn_in=1,
        samples=1,
    )

    # Parent options should capture reduced CPU and request 1 GPU
    assert captured["num_cpus"][0] == 0.25
    assert captured["num_gpus"][0] == 1

