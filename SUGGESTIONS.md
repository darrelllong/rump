# Rump suggestions

> **Motto:** better that, better algorithms
>
> **Creed:** Experiment is asking God for peer review.

2026-09-16 PDT / 2026-09-17 UTC. Current findings, scope and measured evidence:
[AUDIT.md](AUDIT.md). These are proposals, with their acceptance experiments;
no implementation or speedup is claimed by this document. Work from the named
papers, specifications and independently derived mathematics.

## Priorities and acceptance

| Priority | Work | Evidence needed |
|---|---|---|
| 1 | Checked incomplete-beta evaluation | Symmetry and tail fixtures; explicit failure when the chosen expansion cannot certify accuracy |
| 2 | Log-gamma's upper finite boundary | Correct finite/overflow classification around the representability threshold |
| 3 | Reliable consumer checks | Failed graph queries cannot pass; all four consumers checked at identified revisions |
| 4 | Consumer-shaped arithmetic improvements | Lower complete workload cost with exact identities preserved |

## Numerical contracts

For **R1**, separate domain checks, stable normalization, expansion selection,
convergence and output validation. Evaluate small tails directly. A successful
continued-fraction stopping condition must be distinct from exhausting an
iteration budget. A [0,1] range check detects some failures but cannot establish
accuracy: the result 0.2046 at the exact answer 0.5 demonstrates why.

Start from [DLMF §8.17](https://dlmf.nist.gov/8.17), including its symmetry and
continued fractions, and the appropriate large-parameter expansions. Derive the
normalization without subtracting nearly equal large terms. Use `log1p` where
its argument is small. Keep an absolute target near central probabilities and
relative/log-tail targets for rare events; an absolute tolerance alone can erase
a meaningful small probability.

Validate in parameter strata: both small, both large, very unequal, central and
tail x, and either side of each algorithm switch. Include exact
`I_x(1,b)=1-(1-x)^b`, `I_x(a,1)=x^a` and `I_(1/2)(a,a)=1/2`, evaluated with
stable independent formulas. Compare the Student quantiles actually used by
factoring with high-precision values; numerical failure must retain contenders
or stop selection explicitly, never silently discard them.

For **R2**, derive an overflow-safe large-x log-gamma branch from
[DLMF §5.11](https://dlmf.nist.gov/5.11). Add representable neighbors around the
largest x with finite log-gamma, as well as `2.557e305` through `2.559e305`.
Distinguish final-result overflow from an avoidable intermediate overflow.
Preserve the reproducible coefficient construction and the measured small-x
error envelope. A larger sweep complements, rather than replaces, these exact
boundary arguments.

Coordinate with entropy's incomplete-gamma work. A more accurate log-gamma
cannot by itself repair reciprocal overflow or cancellation in a downstream
probability formula.

## Certified search and matrix algebra

The exact Gram–Schmidt and explicit enumeration outcomes are already present.
Extend their tests at the f64 exponent and exact-coefficient limits, with targets
outside the span and integral changes of basis. Compare complete small-lattice
sets with independent bounded enumeration; force low budgets and assert the
outcome as well as the validity of returned vectors. Charge exact setup and
interval work in factoring's actual polynomial-search workload before adding a
more elaborate arithmetic representation.

Preserve `filtered row = XOR(original rows named by its composition)` at every
merge and compaction. Verify expanded dependencies before extraction. For any
new filter policy measure dimensions, nonzeros, composition lengths, expansion
cost and total solve cost. The retained parallel-filter experiment did not
justify adoption; it supplies a baseline and a reason to require a different
workload or mechanism before repeating that proposal.

## Arithmetic cost in real consumers

Profile size and shape separately: balanced versus unbalanced multiplication,
squaring, remainder-only division, repeated moduli, repeated bases and scratch
reuse. Use the existing slow oracles at crossover boundaries. Include allocation,
setup, wiping and destruction in a complete-operation measurement even when a
steady-state microbenchmark deliberately excludes them.

A cryptography-linked graph enables wipe throughout rump; factoring's standalone
graph does not. Record both modes where they matter and avoid transferring a
crossover measured in one to the other without evidence. Retain the measured
scratch/prepared-context results and verify any new choice on consumer inputs.

## Integration evidence

For **R3**, validate query success before interpreting a graph. Test the failure
branch with controlled command failure. Run the full consumer matrix for public
API and numerical-contract changes, including entropy minimal and factoring.
Keep exact revisions and lockfile digests in the result; preserve a fixed
release combination alongside moving integration coverage.

Separate correctness, numerical accuracy, resource behavior and performance in
the record. A benchmark digest proves agreement between its tested outputs;
an independently checked identity is still needed to rule out agreement on the
same wrong answer.
