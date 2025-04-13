import logging
from functools import wraps

# Decorator to suppress gx logging messages
def suppress_gx_logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find all GE-related loggers dynamically
        suppressed = {}
        for name in logging.root.manager.loggerDict:
            if name.startswith("great_expectations"):
                logger = logging.getLogger(name)
                suppressed[name] = logger.level
                logger.setLevel(logging.CRITICAL)
        try:
            return func(*args, **kwargs)
        finally:
            for name, level in suppressed.items():
                logging.getLogger(name).setLevel(level)
    return wrapper


# def suppress_gx_logging(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         loggers = [
#             "great_expectations",
#             "great_expectations.data_context",
#             "great_expectations.datasource",
#             "great_expectations.core",
#             "great_expectations.expectations.expectation",
#             "great_expectations.data_context.data_context.ephemeral",
#             "great_expectations.data_context.data_context",
#             "great_expectations.validator",
#             "great_expectations.validator.validator"
#         ]
#         for l in loggers:
#             logging.getLogger(l).setLevel(logging.CRITICAL)
#         return func(*args, **kwargs)
#     return wrapper


