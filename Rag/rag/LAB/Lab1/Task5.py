class HospitalEnvironment:
    def __init__(self):
        self.layout = {
            "Nurse Station": "Central Hub",
            "Storage Room": "Medicine Storage",
            "Room 101": "Patient Room",
            "Room 102": "Patient Room",
        }
        self.medicine_stock = {"Paracetamol": 4, "Antibiotic": 3}
        self.patient_info = {101: {"id": "P001", "medicine": "Paracetamol"}, 
                             102: {"id": "P002", "medicine": "Antibiotic"}}
        self.schedule = [{"room": 101, "medicine": "Paracetamol"}, 
                         {"room": 102, "medicine": "Antibiotic"}]

class DeliveryRobot:
    def __init__(self, environment):
        self.location = "Nurse Station"
        self.carrying_medicine = None
        self.env = environment

    def move_to(self, destination):
        print(f"Robot moving from {self.location} to {destination}.")
        self.location = destination

    def collect_medicine(self, medicine):
        if self.env.medicine_stock.get(medicine, 0) > 0:
            self.env.medicine_stock[medicine] -= 1
            self.carrying_medicine = medicine
            print(f"Collected {medicine} from storage.")
        else:
            print(f"Medicine {medicine} is out of stock!")

    def deliver_medicine(self, room):
        if self.carrying_medicine:
            print(f"Delivering {self.carrying_medicine} to Room {room}.")
            self.location = f"Room {room}"
            self.carrying_medicine = None
        else:
            print("No medicine to deliver!")

    def verify_patient(self, room):
        patient = self.env.patient_info.get(room, None)
        if patient:
            print(f"Verifying patient in Room {room}. ID: {patient['id']}")
            return True
        else:
            print(f"Patient verification failed in Room {room}. Alerting staff!")
            self.alert_staff(room)
            return False

    def alert_staff(self, room):
        print(f"ALERT: Nurse assistance required in Room {room}!")

hospital = HospitalEnvironment()
robot = DeliveryRobot(hospital)

for task in hospital.schedule:
    target_room = task["room"]
    required_medicine = task["medicine"]

    robot.move_to("Storage Room")
    robot.collect_medicine(required_medicine)

    robot.move_to(f"Room {target_room}")
    if robot.verify_patient(target_room):
        robot.deliver_medicine(target_room)
