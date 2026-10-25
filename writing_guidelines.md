# Writing guidelines for papers

These guidelines come from rewriting *Zero- Versus Infinite-Temperature Damping in Variational Quantum Circuits*. They describe the habits that made the text readable, and the habits that made earlier drafts hard to follow.

## Structure and flow

1. **Open every section with its purpose.** One or two sentences say what the section asks and why ("In this section, we ask how much of the effect of the noise a trainable output scale can undo, and what it costs").
2. **Make each subsection a small story:**
   - the motivation;
   - what we did;
   - what we found, in words;
   - the one or two numbers that matter;
   - what it means;
   - a link to the next part.
3. **Use signposts:** "We now turn to…", "As we show next…", "Having established…, we…", "In summary…".
4. **Put our results first.** Prior work serves as the motivation or as a test case. Credit each prior paper once for each model or result taken from it, not every time it comes up.
5. **Abstract.** Write it in this order: context, gap, "In this work, we…", "We show…", the key numbers, the implication. About 200 words.
6. **Introduction.** About 550 words:
   - context;
   - gap;
   - our results;
   - how they explain earlier reports;
   - a one-sentence outline.

## Sentences

7. **Write in the "we" voice:** "We show", "we find", "we test".
8. **Keep one idea per sentence,** at moderate length. Do not stack clauses with colons and semicolons.
9. **Do not ask rhetorical questions.** Use statements instead: "The floor comes from the non-unital term", not "Why does AD have a floor?"; "To quantify when the two scales separate, we…", not "When do the two scales separate?".
10. **Use plain connectives:** "In contrast", "As a result", "Interestingly", "Note that". Write "compared with", not "against".
11. **Avoid repeated openers,** such as two consecutive sentences that both start with "We".
12. **Introduce terms before shortening them.** Say "the twirled channels" before "the twirls".

## Numbers

13. **Limit each paragraph to the one or two numbers that carry the point.** Move the rest to tables or appendices; do not delete them.
14. **Match the digits to the uncertainty and state units** ("1.5 percentage points", not "1.5").
15. **Make every number traceable to a committed script.** After any rewrite, check that no number was lost.

## Notation and equations

16. **Define every symbol once, where it first appears.** Give each symbol one meaning.
17. **Number only equations that carry a result or are referred to.** Write procedural equations in words. Promote important inline relations to numbered equations.
18. **Keep results and their key steps in the main text, and routine derivations in the appendix.** Punctuate equations as part of the sentence and follow them with "where…".

## Content discipline

19. **Make every passage serve an argument.** Cut related work that does not support a claim.
20. **Do not report claims that are not worth reporting,** such as differences too small to matter.
21. **Explain each thing once.** Remove statements that repeat across sections.
22. **Keep the support in the appendices,** and say in the main text where to find it.

## Process

23. **Revise one section at a time** and review it in the compiled PDF.
24. **Keep a backup before every edit.** After each change, confirm a clean build with no undefined references and no overfull lines.
