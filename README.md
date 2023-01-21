# Perceptron

> Rust implementation of Perceptron

> Check `src/lib.rs` for test case  `cargo test -- --no-capture`

```rust
let data = load_breast_cancer();
let mut percep = Perceptron::new(epoch=1);
percep.fit(&data.records, &data.targets);
```

```
Finished test [unoptimized + debuginfo] target(s) in 0.61s
	Running unittests src/lib.rs (target/debug/deps/perceptron-c790a06670839b80)

running 1 test
accuracy_score=0.6274165202108963
test tests::test_classifier ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```