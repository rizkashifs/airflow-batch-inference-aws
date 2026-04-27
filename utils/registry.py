"""SageMaker Model Registry helpers.

Queries the registry for the latest APPROVED versioned model package and
resolves the metadata needed to run a Batch Transform job against a
*pre-existing* SageMaker Model (no model creation happens here).
"""

from __future__ import annotations

import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def get_latest_approved_model(
    model_package_group_name: str,
    region: str,
    default_instance_type: str = "ml.m5.xlarge",
) -> dict:
    """Return metadata for the latest APPROVED model package in *group*.

    Calls:
        sagemaker.list_model_packages  (paginated, sorted DESC by CreationTime)
        sagemaker.describe_model_package

    Returns a dict with keys:
        model_name        – existing SageMaker Model name (NOT created here)
        model_package_arn – full ARN of the model package
        model_version     – version number as a string
        instance_type     – preferred transform instance type
        creation_time     – ISO-8601 creation timestamp
    """
    client = boto3.client("sagemaker", region_name=region)

    packages = _list_approved_packages(client, model_package_group_name)

    if not packages:
        raise RuntimeError(
            f"No APPROVED model packages found in group '{model_package_group_name}'. "
            "Approve at least one package in the SageMaker Model Registry before running."
        )

    latest = packages[0]
    package_arn: str = latest["ModelPackageArn"]
    model_version: str = str(latest.get("ModelPackageVersion", 1))

    logger.info(
        "Latest approved package — ARN: %s  version: %s  created: %s",
        package_arn,
        model_version,
        latest["CreationTime"].isoformat(),
    )

    description = _describe_package(client, package_arn)
    model_name = _resolve_model_name(description, model_package_group_name)
    instance_type = _resolve_instance_type(description, default_instance_type)

    result = {
        "model_name": model_name,
        "model_package_arn": package_arn,
        "model_version": model_version,
        "instance_type": instance_type,
        "creation_time": latest["CreationTime"].isoformat(),
    }
    logger.info("Resolved model metadata: %s", result)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _list_approved_packages(client, group_name: str) -> list[dict]:
    """Pages through list_model_packages and returns all APPROVED versioned entries."""
    paginator = client.get_paginator("list_model_packages")
    try:
        page_iter = paginator.paginate(
            ModelPackageGroupName=group_name,
            ModelApprovalStatus="Approved",
            ModelPackageType="Versioned",
            SortBy="CreationTime",
            SortOrder="Descending",
        )
        result: list[dict] = []
        for page in page_iter:
            result.extend(page.get("ModelPackageSummaryList", []))
        return result
    except ClientError as exc:
        logger.error(
            "SageMaker list_model_packages failed for group '%s': %s", group_name, exc
        )
        raise


def _describe_package(client, package_arn: str) -> dict:
    try:
        return client.describe_model_package(ModelPackageName=package_arn)
    except ClientError as exc:
        logger.error("SageMaker describe_model_package failed for '%s': %s", package_arn, exc)
        raise


def _resolve_model_name(description: dict, fallback: str) -> str:
    """Derive the SageMaker Model name from a package description.

    Convention: store the deployed model name in CustomerMetadataProperties
    under the key 'sagemaker_model_name' or 'model_name'.
    If absent, falls back to the model package group name (common when a
    single Model is shared across package versions).
    """
    props = description.get("CustomerMetadataProperties") or {}
    for key in ("sagemaker_model_name", "model_name", "ModelName"):
        if key in props and props[key]:
            return props[key]

    group = description.get("ModelPackageGroupName", "").strip()
    return group or fallback


def _resolve_instance_type(description: dict, default: str) -> str:
    """Pick the best transform instance type from InferenceSpecification.

    Preference order:
      1. ml.m5 family (cost-effective for tabular/CSV workloads)
      2. First entry in SupportedTransformInstanceTypes
      3. *default* argument
    """
    spec = description.get("InferenceSpecification") or {}
    types: list[str] = spec.get("SupportedTransformInstanceTypes") or []

    if not types:
        logger.warning(
            "No SupportedTransformInstanceTypes found; using default '%s'.", default
        )
        return default

    for itype in types:
        if itype.startswith("ml.m5"):
            return itype

    return types[0]
