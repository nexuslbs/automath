import time
from env.core import BaseNode
from test_suite import test_root, test_utils

def test():
    start_time = time.time()
    BaseNode.fast = True
    final_states = test_root.test(fast=True)
    amount = test_utils.result_amount(final_states)
    action_amount = test_utils.result_action_amount(final_states)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Amount of completed tests: {amount} ({action_amount} actions)")
    print(f"Time taken to run all tests (fast): {elapsed_time:.2f} seconds")
