

1. **(Easy)** *Rain–Sprinkler–WetGrass*

   * **Structure**:

     ```
     Rain → WetGrass ← Sprinkler
     ```
   * **Tasks**:

     1. Create the `BayesianNetwork([("Rain","WetGrass"),("Sprinkler","WetGrass")])`.
     2. Define:

        * `P(Rain)= [0.2, 0.8]`
        * `P(Sprinkler∣Rain)=[ [0.01,0.99], [0.4,0.6] ]`
        * `P(WetGrass∣Rain,Sprinkler)` as a 2×2 table, e.g. high if either parent true.
     3. Compute with `VariableElimination`:

        * `P(WetGrass=True)`
        * `P(Rain=True ∣ WetGrass=True)`

```     
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define structure
model = BayesianNetwork([
    ('Rain', 'WetGrass'),
    ('Sprinkler', 'WetGrass')
])

# 2. Define CPTs by hand
cpd_rain = TabularCPD(
    variable='Rain', variable_card=2,
    values=[[0.2], [0.8]]
)
# P(Sprinkler | Rain)
# Rain=True → [0.01 spr=on, 0.99 off]; Rain=False → [0.4, 0.6]
cpd_sprinkler = TabularCPD(
    variable='Sprinkler', variable_card=2,
    values=[[0.01, 0.4], [0.99, 0.6]],
    evidence=['Rain'], evidence_card=[2]
)
# P(WetGrass | Rain, Sprinkler)
# Order of states: Rain=[T,F], Sprinkler=[T,F]
cpd_wetgrass = TabularCPD(
    variable='WetGrass', variable_card=2,
    values=[
        # WetGrass = True
        [0.99, 0.9, 0.9, 0.0],
        # WetGrass = False
        [0.01, 0.1, 0.1, 1.0]
    ],
    evidence=['Rain', 'Sprinkler'],
    evidence_card=[2, 2]
)

# 3. Add and validate
model.add_cpds(cpd_rain, cpd_sprinkler, cpd_wetgrass)
assert model.check_model()

# 4. Inference
infer = VariableElimination(model)

# P(WetGrass = True)
q1 = infer.query(['WetGrass'])
print('P(WetGrass=True) =', q1.values[1])

# P(Rain=True | WetGrass=True)
q2 = infer.query(['Rain'], evidence={'WetGrass': 1})
print('P(Rain=True | WetGrass=True) =', q2.values[1])
```


2. **(Moderate)** *Burglary–Earthquake–Alarm–JohnCalls–MaryCalls*

   * **Structure**:

     ```
     Burglary → Alarm ← Earthquake  
     Alarm → JohnCalls  
     Alarm → MaryCalls
     ```
   * **Tasks**:

     1. Build the five‐node network.
     2. Hand‐craft CPTs such as:

        * `P(Burglary)=0.001`
        * `P(Earthquake)=0.002`
        * `P(Alarm∣Burglary,Earthquake)` with four entries
        * `P(JohnCalls∣Alarm)`, `P(MaryCalls∣Alarm)`
     3. Query:

        * `P(Alarm=True ∣ JohnCalls=True, MaryCalls=False)`
       
```
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define structure
model = BayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])

# 2. Define CPTs
cpd_burglary = TabularCPD('Burglary', 2, [[0.001], [0.999]])
cpd_earthquake = TabularCPD('Earthquake', 2, [[0.002], [0.998]])

# Alarm | Burglary, Earthquake
# Order parents: [Burglary, Earthquake] = (TT, TF, FT, FF)
cpd_alarm = TabularCPD(
    'Alarm', 2,
    values=[
        [0.95, 0.94, 0.29, 0.001],  # Alarm=True
        [0.05, 0.06, 0.71, 0.999]   # Alarm=False
    ],
    evidence=['Burglary', 'Earthquake'],
    evidence_card=[2, 2]
)

# JohnCalls | Alarm
cpd_john = TabularCPD(
    'JohnCalls', 2,
    values=[[0.90, 0.05], [0.10, 0.95]],
    evidence=['Alarm'],
    evidence_card=[2]
)

# MaryCalls | Alarm
cpd_mary = TabularCPD(
    'MaryCalls', 2,
    values=[[0.70, 0.01], [0.30, 0.99]],
    evidence=['Alarm'],
    evidence_card=[2]
)

# 3. Add and validate
model.add_cpds(cpd_burglary, cpd_earthquake, cpd_alarm, cpd_john, cpd_mary)
assert model.check_model()

# 4. Inference
infer = VariableElimination(model)

# P(Alarm=True | JohnCalls=True, MaryCalls=False)
q = infer.query(
    ['Alarm'],
    evidence={'JohnCalls': 1, 'MaryCalls': 0}
)
print('P(Alarm=True | J=True, M=False) =', q.values[1])
```



3. **(Challenging)** *Smoking–Cancer–Bronchitis–XRay–Dyspnea*

   * **Structure**:

     ```
     Smoking → Cancer → XRay  
     Smoking → Bronchitis → Dyspnea
     ```
   * **Tasks**:

     1. Define this five‐node DAG.
     2. Manually set CPTs, e.g.:

        * `P(Smoking)=0.3`
        * `P(Cancer∣Smoking)`
        * `P(Bronchitis∣Smoking)`
        * `P(XRay∣Cancer)`
        * `P(Dyspnea∣Bronchitis)`
     3. Compute:

        * `P(Cancer=True ∣ XRay=True, Dyspnea=True)`

```
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define structure
model = BayesianNetwork([
    ('Smoking', 'Cancer'),
    ('Smoking', 'Bronchitis'),
    ('Cancer', 'XRay'),
    ('Bronchitis', 'Dyspnea')
])

# 2. Define CPTs
cpd_smoking = TabularCPD('Smoking', 2, [[0.3], [0.7]])
# Cancer | Smoking
cpd_cancer = TabularCPD(
    'Cancer', 2,
    values=[[0.1, 0.01], [0.9, 0.99]],
    evidence=['Smoking'], evidence_card=[2]
)
# Bronchitis | Smoking
cpd_bronch = TabularCPD(
    'Bronchitis', 2,
    values=[[0.3, 0.05], [0.7, 0.95]],
    evidence=['Smoking'], evidence_card=[2]
)
# XRay | Cancer
cpd_xray = TabularCPD(
    'XRay', 2,
    values=[[0.9, 0.2], [0.1, 0.8]],
    evidence=['Cancer'], evidence_card=[2]
)
# Dyspnea | Bronchitis
cpd_dyspnea = TabularCPD(
    'Dyspnea', 2,
    values=[[0.65, 0.3], [0.35, 0.7]],
    evidence=['Bronchitis'], evidence_card=[2]
)

# 3. Add and validate
model.add_cpds(cpd_smoking, cpd_cancer, cpd_bronch, cpd_xray, cpd_dyspnea)
assert model.check_model()

# 4. Inference
infer = VariableElimination(model)

# P(Cancer=True | XRay=True, Dyspnea=True)
q = infer.query(
    ['Cancer'],
    evidence={'XRay': 1, 'Dyspnea': 1}
)
print('P(Cancer=True | XRay=True, Dyspnea=True) =', q.values[1])
```






4. **(Hard)** *Cloudy–Sprinkler–Rain–WetGrass–Humidity–Traffic*

   * **Structure**:

     ```
     Cloudy → Sprinkler  
     Cloudy → Rain  
     Sprinkler → WetGrass ← Rain  
     Cloudy → Humidity  
     Rain → Traffic
     ```
   * **Tasks**:

     1. Instantiate the six‐node model.
     2. Define CPTs by hand (e.g. make Humidity high when Cloudy, Traffic heavy when Rain, etc.).
     3. Run two inference queries, for example:

        * `P(Rain=True ∣ WetGrass=True, Humidity=High)`
        * `P(Traffic=True ∣ Sprinkler=True, Cloudy=False)`

```
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define structure
model = BayesianNetwork([
    ('Difficulty', 'Grade'),
    ('Intelligence', 'Grade'),
    ('Grade', 'Letter'),
    ('Letter', 'Recommendation')
])

# 2. Define CPTs
cpd_difficulty = TabularCPD('Difficulty', 2, [[0.3], [0.7]])
cpd_intelligence = TabularCPD('Intelligence', 2, [[0.7], [0.3]])

# Grade | Difficulty, Intelligence
# D=H/L, I=H/L → order: (H,H), (H,L), (L,H), (L,L)
cpd_grade = TabularCPD(
    'Grade', 3,
    values=[
        [0.3, 0.05, 0.9, 0.5],  # Grade = A
        [0.4, 0.25, 0.08, 0.3], # Grade = B
        [0.3, 0.7, 0.02, 0.2]   # Grade = C
    ],
    evidence=['Difficulty', 'Intelligence'],
    evidence_card=[2, 2]
)

# Letter | Grade
cpd_letter = TabularCPD(
    'Letter', 2,
    values=[[0.9, 0.7, 0.4], [0.1, 0.3, 0.6]],
    evidence=['Grade'],
    evidence_card=[3]
)

# Recommendation | Letter
cpd_recommend = TabularCPD(
    'Recommendation', 2,
    values=[[0.95, 0.6], [0.05, 0.4]],
    evidence=['Letter'],
    evidence_card=[2]
)

# 3. Add and validate
model.add_cpds(cpd_difficulty, cpd_intelligence, cpd_grade, cpd_letter, cpd_recommend)
assert model.check_model()

# 4. Inference
infer = VariableElimination(model)

# Query: P(Intelligence | Recommendation = Strong)
q = infer.query(['Intelligence'], evidence={'Recommendation': 0})  # 0 = strong
print('P(Intelligence=High | Recommendation=Strong) =', q.values[1])
```


5. **(Ultra‑Hard)** *Weather → Visibility → … → Stress (8 nodes)*

   * **Structure**:

     ```
     Weather → Visibility  
     Weather → Slippery  
     Slippery, Speed, Visibility → Accident  
     Accident → Injury  
     Accident → TrafficDelay → Stress
     ```

     Nodes (all binary):

     * **Weather** (Clear/Rain)
     * **Visibility** (Good/Poor)
     * **Slippery** (No/Yes)
     * **Speed** (Low/High)
     * **Accident** (No/Yes)
     * **Injury** (No/Yes)
     * **TrafficDelay** (No/Yes)
     * **Stress** (Low/High)
   * **Tasks**:

     1. Build the `BayesianNetwork` with the above edges.
     2. Manually specify CPTs for each node—note that **Accident** has three parents (8 entries), **Stress** has one parent, etc.
     3. Verify the model.
     4. Using `VariableElimination`, compute:

        * `P(Accident=True ∣ Speed=High, Weather=Rain, Visibility=Poor)`
        * `P(Injury=True ∣ Stress=High, TrafficDelay=True)`
        * (Bonus) `P(Stress=High ∣ Weather=Rain)`

---


```
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define structure
model = BayesianNetwork([
    ('Genetics', 'Disease'),
    ('Lifestyle', 'Disease'),
    ('Disease', 'Symptom1'),
    ('Disease', 'Symptom2'),
    ('Disease', 'Test1'),
    ('Disease', 'Test2'),
    ('Disease', 'Treatment')
])

# 2. Define CPTs
cpd_genetics = TabularCPD('Genetics', 2, [[0.1], [0.9]])
cpd_lifestyle = TabularCPD('Lifestyle', 2, [[0.3], [0.7]])

# Disease | Genetics, Lifestyle
cpd_disease = TabularCPD(
    'Disease', 2,
    values=[
        [0.95, 0.8, 0.6, 0.1],  # Disease = True
        [0.05, 0.2, 0.4, 0.9]   # Disease = False
    ],
    evidence=['Genetics', 'Lifestyle'],
    evidence_card=[2, 2]
)

# Symptom1 | Disease
cpd_symptom1 = TabularCPD(
    'Symptom1', 2,
    values=[[0.8, 0.1], [0.2, 0.9]],
    evidence=['Disease'], evidence_card=[2]
)

# Symptom2 | Disease
cpd_symptom2 = TabularCPD(
    'Symptom2', 2,
    values=[[0.7, 0.2], [0.3, 0.8]],
    evidence=['Disease'], evidence_card=[2]
)

# Test1 | Disease
cpd_test1 = TabularCPD(
    'Test1', 2,
    values=[[0.9, 0.2], [0.1, 0.8]],
    evidence=['Disease'], evidence_card=[2]
)

# Test2 | Disease
cpd_test2 = TabularCPD(
    'Test2', 2,
    values=[[0.85, 0.3], [0.15, 0.7]],
    evidence=['Disease'], evidence_card=[2]
)

# Treatment | Disease
cpd_treatment = TabularCPD(
    'Treatment', 2,
    values=[[0.95, 0.1], [0.05, 0.9]],
    evidence=['Disease'], evidence_card=[2]
)

# 3. Add and validate
model.add_cpds(cpd_genetics, cpd_lifestyle, cpd_disease, cpd_symptom1,
               cpd_symptom2, cpd_test1, cpd_test2, cpd_treatment)
assert model.check_model()

# 4. Inference
infer = VariableElimination(model)

# Query: P(Disease | Test1=Positive, Test2=Positive, Symptom1=Present, Symptom2=Present)
q = infer.query(['Disease'], evidence={'Test1': 0, 'Test2': 0, 'Symptom1': 0, 'Symptom2': 0})
print('P(Disease=True | Positive tests, Present symptoms) =', q.values[1])
```
