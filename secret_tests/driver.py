import importlib.util
import datetime
import os
import numpy as np
import random
import contextlib
from io import StringIO

@contextlib.contextmanager
def suppress_output():
    with contextlib.redirect_stdout(StringIO()):
        yield

def test_student_code(solution_path):
    report_dir = os.path.join(os.path.dirname(__file__), "..", "student_workspace")
    report_path = os.path.join(report_dir, "report.txt")
    os.makedirs(report_dir, exist_ok=True)

    spec = importlib.util.spec_from_file_location("student_module", solution_path)
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)

    report_lines = [f"\n=== HeartRateAnalyzer Test Run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="]
    failures = set()

    # -------------------- ANTI-CHEAT TESTS --------------------
    try:
        random_input = [random.randint(45, 160) for _ in range(5)]
        expected = np.array(random_input)
        with suppress_output():
            result = student_module.create_heart_rate_array(random_input)
        if not np.array_equal(result, expected):
            failures.add("create_heart_rate_array")
    except:
        failures.add("create_heart_rate_array")

    try:
        test_array = np.array([random.randint(40, 180) for _ in range(10)])
        with suppress_output():
            valid = student_module.validate_heart_rate_array(test_array)
        if valid is not True:
            failures.add("validate_heart_rate_array")
    except:
        failures.add("validate_heart_rate_array")

    try:
        arr = np.array([50, 60, 70, 110])
        expected = (round(np.mean(arr)), np.max(arr), np.min(arr))
        with suppress_output():
            result = student_module.compute_health_metrics(arr)
        if result != expected:
            failures.add("compute_health_metrics")
    except:
        failures.add("compute_health_metrics")

    try:
        arr = np.array([55, 60, 85, 101])
        expected = ["Bradycardia", "Normal", "Normal", "Tachycardia"]
        with suppress_output():
            result = student_module.detect_abnormal_rates(arr)
        if not np.array_equal(result, expected):
            failures.add("detect_abnormal_rates")
    except:
        failures.add("detect_abnormal_rates")

    try:
        arr = np.array([60, 62, 65, 58, 100, 59, 102, 101])
        with suppress_output():
            streak = student_module.longest_normal_streak(arr)
        if streak < 3:
            failures.add("longest_normal_streak")
    except:
        failures.add("longest_normal_streak")

    try:
        arr = np.array([66, 77])
        expected = np.array(["66.00 BPM", "77.00 BPM"])
        with suppress_output():
            result = student_module.format_heart_rate_readings(arr)
        if not np.array_equal(result, expected):
            failures.add("format_heart_rate_readings")
    except:
        failures.add("format_heart_rate_readings")

    # -------------------- MAIN TEST CASES --------------------

    test_cases = [
        ("Visible", {
            "id": "TC1", "desc": "Creating a valid heart rate array",
            "func": "create_heart_rate_array", "input": [72, 85, 60],
            "expected": np.array([72, 85, 60])
        }),
        ("Visible", {
            "id": "TC2", "desc": "Validating out-of-range BPM values",
            "func": "validate_heart_rate_array", "input": np.array([72, 85, 60, 200]),
            "expected_bool": False
        }),
        ("Visible", {
            "id": "TC3", "desc": "Computing average, max, and min BPM",
            "func": "compute_health_metrics",
            "input": np.array([72, 85, 60, 98, 105]),
            "expected": (84, 105, 60)
        }),
        ("Visible", {
            "id": "TC4", "desc": "Detecting abnormal heart rate conditions",
            "func": "detect_abnormal_rates",
            "input": np.array([72, 85, 60, 98, 105]),
            "expected": np.array(["Normal", "Normal", "Normal", "Normal", "Tachycardia"])
        }),
        ("Visible", {
            "id": "TC5", "desc": "Longest normal streak",
            "func": "longest_normal_streak",
            "input": np.array([72, 85, 60, 98, 105, 80, 75, 72]),
            "expected_int": 4
        }),
        ("Hidden", {
            "id": "HTC1", "desc": "Formatting BPM readings with two decimals",
            "func": "format_heart_rate_readings",
            "input": np.array([72, 85]),
            "expected": np.array(["72.00 BPM", "85.00 BPM"])
        }),
        ("Hidden", {
            "id": "HTC2", "desc": "Handling empty array",
            "func": "validate_heart_rate_array",
            "input": np.array([]), "expected_bool": False
        }),
        ("Hidden", {
            "id": "HTC3", "desc": "Borderline BPM classifications",
            "func": "detect_abnormal_rates",
            "input": np.array([59, 60, 100, 101]),
            "expected": np.array(["Bradycardia", "Normal", "Normal", "Tachycardia"])
        }),
    ]

    for section, case in test_cases:
        try:
            func = getattr(student_module, case["func"])
            with suppress_output():
                result = func(case["input"]) if not isinstance(case["input"], tuple) else func(*case["input"])

            if case["func"] in failures:
                msg = f"❌ {case['id']}: {case['desc']} failed | Reason: Logic violation / hardcoded output"
            else:
                if "expected" in case:
                    assert np.array_equal(result, case["expected"])
                elif "expected_bool" in case:
                    assert result == case["expected_bool"]
                elif "expected_int" in case:
                    assert result == case["expected_int"]
                msg = f"✅ {case['id']}: {case['desc']}"
        except Exception as e:
            msg = f"❌ {case['id']}: {case['desc']} failed | Reason: {str(e)}"

        print(msg)
        report_lines.append(msg)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

if __name__ == "__main__":
    test_student_code()
