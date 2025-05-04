class Environment:
    def __init__(self):
        self.hospital_layout = {
            "Nurse Station": "Starting Point",
            "Medicine Storage": "Storage Area",
            "Room 101": "Patient Room",
            "Room 205": "Patient Room",
        }
        self.patient_schedule = [
            {"room": 101, "medicine": "Medicine A", "time": "10:00 AM"},
            {"room": 205, "medicine": "Medicine B", "time": "11:00 AM"},
        ]
        self.medicine_storage = {
            "Medicine A": 5,
            "Medicine B": 3,
        }
        self.patient_Id = {101: "Patient_1", 205: "Patient_2"}

class Agent:
    def __init__(self):
        self.current_loc = "Nurse Station"
        self.picked_med = None

    def Move(self, station):
        print(f"Moving from {self.current_loc} to {station}")
        self.current_loc = station

    def PickUpMedicine(self, medicine, env):
        if medicine in env.medicine_storage and env.medicine_storage[medicine] > 0:
            print(f"Picked up {medicine}")
            env.medicine_storage[medicine] -= 1  
            self.current_loc = "Medicine Storage"
            self.picked_med = medicine
        else:
            print(f"Medicine {medicine} not available!")

    def DeliverMedicine(self, room):
        if self.picked_med:
            print(f"Delivering {self.picked_med} to patient in Room {room}")
            self.current_loc = f"Room {room}"
            self.picked_med = None  
        else:
            print("No medicine picked up!")

    def ScanPatientId(self, patient_id, room, env):
        if room in env.patient_Id and env.patient_Id[room] == patient_id:
            self.DeliverMedicine(room)
        else:
            print(f"Patient ID {patient_id} does not match Room {room}")
            self.AlertStaff(room)

    def AlertStaff(self, room):
        print(f"ALERT! Nurse needed in Room {room}")

    def run(self, env):

        for task in env.patient_schedule:
            room = task["room"]
            medicine = task["medicine"]

            self.Move("Medicine Storage")
            self.PickUpMedicine(medicine, env)

            self.Move(f"Room {room}")

            patient_id = env.patient_Id.get(room, None)
            if patient_id:
                self.ScanPatientId(patient_id, room, env)
            else:
                print(f"No registered patient ID for Room {room}")
                self.AlertStaff(room)


env = Environment()
agent = Agent()
agent.run(env)
