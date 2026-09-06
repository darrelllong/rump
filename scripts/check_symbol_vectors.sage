# Recomputes every Jacobi and Kronecker test vector in src/number_theory.rs
# with SageMath and reports any mismatch. The vectors are the crate's own
# data — kept in the source so the tests need no Sage — and this is how
# they are regenerated or audited: `sage scripts/check_symbol_vectors.sage`.
import re
source = open("src/number_theory.rs").read()

def vectors(name):
    start = source.index("const %s: &[(&str, &str, i8)] = &[" % name)
    end = source.index("];", start)
    return re.findall(r'\("([0-9a-f]+)", "([0-9a-f]+)", (-?\d)\)', source[start:end])

bad = 0
for a, n, expected in vectors("JACOBI_VECTORS"):
    got = jacobi_symbol(Integer(a, 16), Integer(n, 16))
    if got != Integer(expected):
        bad += 1
        print("jacobi", a, n, "expected", expected, "sage", got)
for a, n, expected in vectors("KRONECKER_VECTORS"):
    got = kronecker_symbol(Integer(a, 16), Integer(n, 16))
    if got != Integer(expected):
        bad += 1
        print("kronecker", a, n, "expected", expected, "sage", got)
print("checked", len(vectors("JACOBI_VECTORS")) + len(vectors("KRONECKER_VECTORS")), "mismatches", bad)
