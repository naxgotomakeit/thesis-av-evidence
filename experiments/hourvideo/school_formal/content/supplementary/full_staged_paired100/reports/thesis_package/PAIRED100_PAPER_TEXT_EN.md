# R3 Full Staged vs. R3 Direct on the paired-100 subset

## Experimental setting

We compared frozen R3 Full Staged and R3 Direct-v1.2 outputs on the same 100 questions drawn from 12 videos. Full Staged reused the frozen R3 Planner outputs and newly executed the Shared, Fine, and Final Haiku stages. Direct used the previously frozen R3 Direct routes. All accuracy and completion figures use a fixed denominator of 100; failed routes were retained and counted as incorrect. The paired population, runtime, and raw result closure are identified by manifest SHA-256 `963c3c30dfbe73d7602a54558fa866cc0f74ddad35e12ee7c3fb268f78cacf8b`, runtime fingerprint `06dc90fb8652a8a8cd392bf654040c64446fca246a1bc800d2b45a53a40820e2`, and raw-closure SHA-256 `fec945bfcf788b4cdf3b709794e6797e2a5bf974c83005cb86ef58b2bcdc3a17`, respectively.

## Results

Full Staged answered 24/100 questions correctly (24%) and produced valid predictions for 97/100 routes. R3 Direct answered 32/100 correctly (32%) and produced predictions for 99/100 routes. Fifteen questions were answered correctly by both methods, 59 by neither, 17 by Direct only, and 9 by Full Staged only. A two-sided exact McNemar test on the 26 discordant pairs gave p=0.1686375439. Thus, this paired subset does not provide conventional 0.05-level evidence for a difference, but the test also does not establish equivalence.

The new Shared/Fine/Final execution required 478 logical stage calls and 662 physical provider requests, including 184 validation retries and no transport retries. It transmitted 1,424 image instances when validation retransmissions were counted and accumulated 1,312 within-question unique reviewed Fine images (13.12 per question). Its cache-aware API cost was $11.195644600. The selected historical Planner records contributed an estimated $3.299245000, giving a reconstructed Full Staged cost of $14.494889600. Direct used 487 model turns/physical requests, 928 new unique image transports (9.28 per question), and cost $4.301863350. Relative to Direct and using unrounded values, the new downstream portion cost 160.251% more, while the reconstructed Planner-plus-downstream cost was 236.944% higher.

The measured downstream API-latency sum was 6432.660 s, and the sum of per-route wall times was 6511.081 s (mean 65.111, median 41.127, P90 136.903 s). Direct recorded 1043.289 s of API latency and 1116.593 s of route wall time (mean 11.166, median 9.796, P90 16.852 s). Planner and downstream timings came from separate historical runs; their additive reconstruction must not be described as a measured end-to-end wall clock.

## Retry-contract clarification

The frozen setting `max_validation_retries=2` means two retries after the initial response, not two total attempts. The executed V6.6.2 code therefore allowed up to three validation attempts for Final. Two failed routes each recorded three provider-envelope-valid Final responses that were rejected by the scientific validator because `answer_text` did not match the selected option text. An older protocol-review sentence claiming that Final allowed two attempts was a documentation error; the frozen runtime, manifest, and formal logs consistently show the three-attempt rule. A third Full Staged route failed with a post-validation `KeyError: 'uncertainty'`. The historical Direct comparison contained one turn-limit failure.

## Limitations

Questions from the same video are not independent, whereas the item-level McNemar calculation treats pairs as independent. The p-value should therefore be interpreted descriptively and not as a cluster-aware confirmatory test. The paired-100 subset is not the complete Eval300 population. Full Staged and Direct also differ in inference topology and visual-budget semantics: Full Staged permits up to 16 images per Fine batch and can retransmit them during validation retries, whereas Direct has a global ceiling of 16 unique images per question. Finally, historical Planner usage did not retain the same cache-token breakdown as the downstream run, and reconstructed cross-run latency is not a measured end-to-end runtime.
