from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Step 1: Define the structure of the Bayesian Network
model = DiscreteBayesianNetwork(
    [
        ('Intelligence','Grade'),
        ('StudyHours','Grade'),
        ('Difficulty','Grade'),
        ('Grade','Pass')
    ]
)

cpd_Intelligence = TabularCPD(variable='Intelligence',variable_card=2,
                              values=[[0.7], [0.3]],state_names={'Intelligence': ['High', 'Low']})
cpd_StudyHours = TabularCPD(variable='StudyHours',variable_card=2,
                            values=[[0.6], [0.4]],state_names={'StudyHours':['Sufficient','Insufficient']})
cpd_Difficulty = TabularCPD(variable='Difficulty',variable_card=2,
                            values=[[0.4], [0.6]],state_names={'Difficulty': ['Hard', 'Easy']})
cpd_Grade = TabularCPD(variable='Grade',variable_card=3,
                       values=[
                            [0.9, 0.7, 0.5, 0.3, 0.6, 0.2, 0.1, 0.05],  # A
                            [0.08, 0.2, 0.3, 0.4, 0.3, 0.5, 0.3, 0.25], # B
                            [0.02, 0.1, 0.2, 0.3, 0.1, 0.3, 0.6, 0.7]   # C
                       ],
                       evidence=['Intelligence', 'StudyHours', 'Difficulty'],
                       evidence_card=[2, 2, 2],
                       state_names={
                            'Grade': ['A', 'B', 'C'],
                            'Intelligence': ['High', 'Low'],
                            'StudyHours': ['Sufficient', 'Insufficient'],
                            'Difficulty': ['Hard', 'Easy']
                       })
cpd_pass = TabularCPD(variable='Pass',variable_card=2,
                        values=[
                            [0.95, 0.8, 0.5],
                            [0.05, 0.2, 0.5]   
                        ],
                        evidence=['Grade'],
                        evidence_card=[3],
                        state_names={
                            'Pass': ['Yes', 'No'],
                            'Grade': ['A', 'B', 'C']
                        })

model.add_cpds(cpd_Intelligence,cpd_StudyHours,cpd_Difficulty,cpd_Grade,cpd_pass)

assert model.check_model()

inference = VariableElimination(model)

result_pass = inference.query(
    variables=['Pass'],
    evidence={'StudyHours': 'Sufficient', 'Difficulty': 'Hard'},
    show_progress=False
)
print("P(Pass | StudyHours = Sufficient, Difficulty = Hard):")
print(result_pass)

result_intel = inference.query(
    variables=['Intelligence'],
    evidence={'Pass': 'Yes'},
    show_progress=False
)
print("\nP(Intelligence | Pass = Yes):")
print(result_intel)
