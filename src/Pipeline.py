import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor, ScriptProcessor
from sagemaker import Model
from sagemaker.xgboost import XGBoostPredictor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig, TuningStep
from sagemaker.workflow.step_collections import RegisterModel, CreateModelStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
    WarmStartConfig,
    WarmStartTypes,
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_pipeline(
    region,
    role,
    default_bucket,
    model_package_group_name,
    base_job_prefix,
    pipeline_name,
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    print("Base Directory :  ",BASE_DIR)
# Define variables and parameters needed for the Pipeline steps
    #region = sagemaker.Session().boto_region_name

    sm_client = boto3.client("sagemaker")
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.session.Session(boto_session=boto_session, sagemaker_client=sm_client,default_bucket=default_bucket)

    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://sagemaker-servicecatalog-seedcode-{region}/dataset/abalone-dataset.csv",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    # Cache Pipeline steps to reduce execution time on subsequent executions
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="PreprocessAbaloneDataForHPO",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        cache_config=cache_config,
    )

    model_path = f"s3://{default_bucket}/{base_job_prefix}/AbaloneTrain"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/abalone-train",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    xgb_train.set_hyperparameters(
        eval_metric="rmse",
        objective="reg:squarederror",  # Define the object metric for the training job
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )

    objective_metric_name = "validation:rmse"

    hyperparameter_ranges = {
        "alpha": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
        "lambda": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
    }

    tuner_log = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=3,
        max_parallel_jobs=3,
        strategy="Random",
        objective_type="Minimize",
    )

    step_tuning = TuningStep(
        name="HPTuning",
        tuner=tuner_log,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    parent_tuning_job_name = (
        step_tuning.properties.HyperParameterTuningJobName
    )  # Use the parent tuning job specific to the use case

    warm_start_config = WarmStartConfig(
        WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM, parents={parent_tuning_job_name}
    )

    tuner_log_warm_start = HyperparameterTuner(
        xgb_train,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=3,
        max_parallel_jobs=3,
        strategy="Random",
        objective_type="Minimize",
        warm_start_config=warm_start_config,
    )

    step_tuning_warm_start = TuningStep(
        name="HPTuningWarmStart",
        tuner=tuner_log_warm_start,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    model_bucket_key = f"{default_bucket}/{base_job_prefix}/AbaloneTrain"
    best_model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
        sagemaker_session=sagemaker_session,
        role=role,
        predictor_cls=XGBoostPredictor,
    )

    step_create_first = CreateModelStep(
        name="CreateTopModel",
        model=best_model,
        inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.m4.large"),
    )

    second_best_model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(top_k=1, s3_bucket=model_bucket_key),
        sagemaker_session=sagemaker_session,
        role=role,
        predictor_cls=XGBoostPredictor,
    )

    step_create_second = CreateModelStep(
        name="CreateSecondBestModel",
        model=second_best_model,
        inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.m4.large"),
    )

    # A ProcessingStep is used to evaluate the performance of a selected model from the HPO step. In this case, the top performing model
    # is evaluated. Based on the results of the evaluation, the model is registered into the Model Registry using a ConditionStep.

    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-tuning-step-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    evaluation_report = PropertyFile(
        name="BestTuningModelEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    # This can be extended to evaluate multiple models from the HPO step
    step_eval = ProcessingStep(
        name="EvaluateTopModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    step_register_best = RegisterModel(
        name="RegisterBestAbaloneModel",
        estimator=xgb_train,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=model_bucket_key),
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
    )

    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step=step_eval, property_file=evaluation_report, json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckMSEAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register_best],
        else_steps=[],
    )

    pipeline = Pipeline(
        name= pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            input_data,
            model_approval_status,
        ],
        steps=[step_process, step_tuning, step_create_first, step_create_second, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


