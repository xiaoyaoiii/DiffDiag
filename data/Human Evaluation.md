## Human Evaluation

To evaluate the reliability of the simulation pipeline, three human annotators independently reviewed 100 randomly sampled dialogue turns. Each sampled dialogue turn contained the outputs of the doctor agent, patient agent, and judge model. The annotators checked whether each component followed its corresponding role-specific instructions and whether the generated outputs were faithful to the given input information.

Disagreements among annotators were resolved by majority vote. Inter-annotator agreement was measured using Fleiss' kappa.

### Evaluation Criteria

#### Doctor Agent

The doctor agent was evaluated based on whether it correctly followed the next-step decision provided by the system.

An output was marked as correct if it satisfied all of the following conditions:

1. If the system provided a list of symptoms to ask about, the doctor asked only about the specified symptom or symptoms.
2. If multiple symptoms were provided, the doctor could ask about them one by one or combine them naturally into a single question.
3. If no further symptom inquiry was needed and a final diagnosis was provided, the doctor directly output the diagnosis.
4. The doctor did not ask about symptoms outside the provided symptom list.
5. The doctor did not introduce unsupported symptoms, diseases, medical evidence, or reasoning.

An output was marked as incorrect if the doctor agent asked about symptoms that were not provided, changed the meaning of the queried symptom, omitted a symptom that should have been asked, failed to provide the diagnosis when a final diagnosis was required, or added unsupported medical information.

#### Patient Agent

The patient agent was evaluated based on whether its response was consistent with the gold symptom information.

An output was marked as correct if it satisfied the following rules:

1. If the queried symptom appeared in the positive symptom list, the patient response should clearly confirm the presence of that symptom.
2. If the queried symptom appeared in the negative symptom list, the patient response should clearly deny the presence of that symptom.
3. If the queried symptom appeared in neither the positive nor the negative symptom list, the patient response should express uncertainty, lack of knowledge, or that the symptom had not been noticed.
4. The response should not introduce unsupported symptoms or contradict the provided symptom lists.

An output was marked as incorrect if it contradicted the gold symptom status, introduced symptoms not provided in the input, failed to answer the doctor's question, or incorrectly changed an unknown symptom into a clearly positive or negative one.

#### Judge Model

The judge model was the main focus of the reliability evaluation. The goal was to examine whether the model could correctly convert each doctor-patient dialogue turn into structured symptom-status pairs.

The input to the judge model included the doctor's question, the patient's response, and the queried symptom list. Its output should contain only the queried symptoms and assign each symptom one of three statuses: `positive`, `negative`, or `unknown`.

A judge-model output was marked as correct only if it satisfied all of the following conditions:

1. **Complete symptom coverage**  
   The output generated one symptom-status pair for each queried symptom.

2. **No omission**  
   The output did not miss any queried symptom.

3. **No hallucination**  
   The output did not contain any symptom outside the queried symptom list.

4. **Exact symptom name match**  
   The output used the original symptom names from the queried symptom list and did not rewrite, merge, split, or replace them with related symptoms.

5. **Correct positive and negative status assignment**  
   A symptom was labeled as `positive` only when the patient clearly confirmed its presence. A symptom was labeled as `negative` only when the patient clearly denied its presence.

6. **Correct unknown status assignment**  
   A symptom was labeled as `unknown` when the patient expressed uncertainty, did not know, had not noticed the symptom, or gave an irrelevant or insufficient answer.

7. **Valid output format**  
   The output followed the required structured format, such as a JSON object containing a list of symptom-status pairs.

A judge-model output was marked as incorrect if it assigned an incorrect status, omitted a queried symptom, added an unqueried symptom, modified a symptom name, made an unsupported inference, or produced an invalid output format.
