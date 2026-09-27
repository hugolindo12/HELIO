from main import calculate_total, apply_discount

def test_calculate():
    assert calculate_total([10, 20, 30]) == 60

def test_discount():
    assert abs(apply_discount(60, 10) - 54.0) < 0.01
