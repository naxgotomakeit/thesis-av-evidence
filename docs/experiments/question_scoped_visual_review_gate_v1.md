# Question-scoped visual review gate v1

This isolated adapter prevents supplementary requirements that are not asked by
the current question from reaching the visual-review router or final-answer
handoff. It does not alter Organizer maps, Retrieval results, Fine bindings, or
historical outputs.

Requirements are separated into answer-required, evidence-support-only, and
supplementary-not-requested scopes. A fourth retained-without-refinement scope
keeps context that is relevant at a coarse category level while preventing an
over-specific old detail request from triggering images. Only answer-required
and evidence-support-only scopes can trigger local visual review. In particular,
a question about weapon visibility does not ask for the holder's rank, and a
question about whether medical assistance occurred does not ask for the
provider's name, rank, unit, or the recipient's exact identity.
