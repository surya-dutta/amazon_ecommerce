from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

class CustomETLOperator(BaseOperator):
    @apply_defaults
    def __init__(self, my_param, *args, **kwargs):
        super(CustomETLOperator, self).__init__(*args, **kwargs)
        self.my_param = my_param

    def execute(self, context):
        #print(f"Executing with parameter: {self.my_param}")
        data = pd.read_csv(self.source_path)
        transformed_data = clean_text(data)
        # Save the transformed data
        transformed_data.to_csv(self.destination_path, index=False)
        self.log.info(f"Data transformed and saved to {self.destination_path}")
