def print_settings_line_by_line(settings_obj) -> None:
#     # pydantic v2
#     if hasattr(settings_obj, "model_dump"):
#         data = settings_obj.model_dump()
#     # pydantic v1
#     elif hasattr(settings_obj, "dict"):
#         data = settings_obj.dict()
#     else:
#         data = vars(settings_obj)

#     for k, v in data.items():
#         print(f"{k} = {v}")