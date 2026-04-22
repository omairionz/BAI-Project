#course_tools.py

PREREQUISITES = {
    "CS2100": ["CS1110"],
    "CS2120": ["CS1110"],
    "CS2130": ["CS1110"],
    "CS3100": ["CS2100", "CS2120"],
    "CS3120": ["CS3100"],
    "CS3130": ["CS2130", "CS2100"],
    "CS3140": ["CS2100"]

}
BS_REQUIRED ={"CS3240"}

BA_REQUIRED = set()

BA_CS_ELECTIVES_COUNT = 3
BS_CS_ELECTIVES_COUNT = 5
BA_INTEGRATION_ELECTIVES = 12
BS_INTEGRATION_ELECTIVES = 0
BS_CAPSTONE_REQUIRED = True
BA_CAPSTONE_REQUIRED = False

def check_prereqs(course_code, completed_courses):
    required = PREREQUISITES.get(course_code, [])
    missing = [c for c in required if c not in completed_courses]

    return{
        "course": course_code,
        "eligible": len(missing) == 0,
        "missing_prereqs": missing,
        "completed_all": len(missing)==0
    }
def check_degree_requirements(completed_courses, degree_type):
    degree_type = degree_type.upper().strip()
    if degree_type == "BA":
        required = BA_REQUIRED
    elif degree_type == "BS":
        required = BS_REQUIRED
    else:
        return{
            "error": "Invalid degree type. Must be 'BA' or 'BS'."
        }
    missing = [c for c in required if c not in completed_courses]
    return {
        "degree_type": degree_type,
        "eligible": len(missing) == 0,
        "missing_required_courses": missing,
        "completed_all_required": len(missing) == 0
    }
COURSE_DIFFICULTY = {
    "CS1110": "Easy",
    "CS2100": "Medium",
    "CS2120": "Medium",
    "CS2130": "Medium",
    "CS3100": "Hard",
    "CS3120": "Hard",
    "CS3130": "Hard",
    "CS3140": "Hard",
}
def get_course_difficulty(course_code):
    return {
        "course": course_code,
        "difficulty": COURSE_DIFFICULTY.get(course_code, "Unknown")
    }
