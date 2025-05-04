from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Step 1: Define the structure of the Bayesian Network
model = DiscreteBayesianNetwork(
    [
        ('Disease','Fever'),
        ('Disease','Cough'),
        ('Disease','Fatigue'),
        ('Disease','Chills')
    ]
)

cpd_Disease = TabularCPD(variable='Disease',variable_card=2,
                              values=[[0.3], [0.7]],state_names={'Disease': ['Flu', 'Cold']})
cpd_Fever = TabularCPD(variable='Fever',variable_card=2,
                            values=[[0.9, 0.5],
                                    [0.1, 0.5]],state_names={'Fever':['Yes','No'],'Disease':['Flu','Cold']},
                            evidence=['Disease'],evidence_card=[2])
cpd_Cough = TabularCPD(variable='Cough',variable_card=2,
                            values=[[0.8, 0.6], 
                                    [0.2, 0.4]],state_names={'Cough': ['Yes', 'No'],'Disease':['Flu','Cold']},
                            evidence=['Disease'],evidence_card=[2])
cpd_Fatigue = TabularCPD(variable='Fatigue',variable_card=2,
                       values=[
                            [0.7, 0.3],
                            [0.3,0.7]
                       ],
                       evidence=['Disease'],
                       evidence_card=[2],
                       state_names={
                            'Fatigue':['Yes','No'],
                            'Disease':['Flu','Cold']
                       })
cpd_Chills = TabularCPD(variable='Chills',variable_card=2,
                        values=[
                            [0.6, 0.4],
                            [0.4, 0.6]
                        ],
                        evidence=['Disease'],
                        evidence_card=[2],
                        state_names={
                            'Chills': ['Yes', 'No'],
                            'Disease': ['Flu','Cold']
                        })

model.add_cpds(cpd_Disease,cpd_Fever,cpd_Cough,cpd_Fatigue,cpd_Chills)

assert model.check_model()

inference = VariableElimination(model)

result_t1 = inference.query(
    variables=['Disease'],
    evidence={'Fever': 'Yes', 'Cough': 'Yes'},
    show_progress=False
)
print("Inference Task 1: P(Disease | Fever=Yes, Cough=Yes)")
print(result_t1)

result_t2 = inference.query(
    variables=['Disease'],
    evidence={'Fever': 'Yes', 'Cough': 'Yes','Chills':'Yes'},
    show_progress=False
)
print("\nInference Task 2: P(Disease | Fever=Yes, Cough=Yes, Chills=Yes)")
print(result_t2)

result_t3 = inference.query(
    variables=['Fatigue'],
    evidence={'Disease':'Flu'},
    show_progress=False
)

print("\nInference Task 3: P(Fatigue=Yes | Disease=Flu)")
print(result_t3)
