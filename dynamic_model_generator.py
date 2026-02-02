
from pydantic import create_model, Field

def create_dynamic_output_model(field_names: list[str]):
    """
    Dynamically creates a Pydantic model based on the provided field names.
    Each field is optional (can be None) and has a string type.
    """
    fields = {
        name.strip().replace(" ", "_"): (str | None, Field(default=None, description=f"The {name} of the person"))
        for name in field_names
        if name.strip()
    }
    return create_model('DynamicOutput', **fields)
