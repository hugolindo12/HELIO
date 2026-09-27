"""Demo project."""

def calculate_total(prices):
    total = 0
    for p in prices:
        total += p
    return total

def apply_discount(total, percent):
    return total * percent / 100

if __name__ == "__main__":
    prices = [10, 20, 30]
    total = calculate_total(prices)
    final = apply_discount(total, 10)
    print(f"Total: {total}, After discount: {final}")
    assert abs(final - 54.0) < 0.01, f"Expected 54.0, got {final}"
