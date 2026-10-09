import time
from test_suite import test_root, test_utils

def test():
    start_time = time.time()
    final_states = test_root.test()
    amount = test_utils.result_amount(final_states)
    action_amount = test_utils.result_action_amount(final_states)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Amount of completed tests: {amount} ({action_amount} actions)")
    print(f"Time taken to run all tests: {elapsed_time:.2f} seconds")
