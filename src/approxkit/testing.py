from timeit import default_timer as timer


def test_docstrings():
    import doctest
    print("Running doctests...")

    t0 = timer()
    result = doctest.testmod(
        optionflags=(doctest.NORMALIZE_WHITESPACE
                     | doctest.ELLIPSIS
        )
    )
    dt = timer() - t0

    print(
        f"Attempted: {result.attempted}, "
        f"Failed: {result.failed}, "
        f"Elapsed: {dt:.3f}s"
    )
    return result