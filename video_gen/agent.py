import asyncio
import logging
import os
import time
import uuid

from collections.abc import AsyncIterable, AsyncGenerator
from typing import Any, ClassVar, List
from urllib.parse import urlparse

import google.auth

from google import genai
from google.cloud import storage
from google.genai import types as genai_types
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions


logger = logging.getLogger(__name__)


class VideoGenerationAgent(BaseAgent):
    """An agent that generates video from a text prompt using VEO,
    providing periodic updates and a final GCS URL for the video.
    """

    SUPPORTED_INPUT_CONTENT_TYPES: ClassVar[List[str]] = ['text', 'text/plain']
    SUPPORTED_OUTPUT_CONTENT_TYPES: ClassVar[List[str]] = ['text/plain', 'video/mp4']

    VEO_MODEL_NAME: ClassVar[str] = os.getenv('VEO_MODEL_NAME', 'veo-2.0-generate-001')
    VEO_POLLING_INTERVAL_SECONDS: ClassVar[int] = int(
        os.getenv('VEO_POLLING_INTERVAL_SECONDS', '5')
    )
    VEO_SIMULATED_TOTAL_GENERATION_TIME_SECONDS: ClassVar[int] = int(
        os.getenv('VEO_SIMULATED_TOTAL_GENERATION_TIME_SECONDS', '120')
    )  # 2 minutes for simulated progress
    VEO_DEFAULT_PERSON_GENERATION: ClassVar[str] = 'dont_allow'
    VEO_DEFAULT_ASPECT_RATIO: ClassVar[str] = '16:9'

    GCS_BUCKET_NAME_ENV_VAR: ClassVar[str] = 'VIDEO_GEN_GCS_BUCKET'
    SIGNED_URL_EXPIRATION_SECONDS: ClassVar[int] = 3600 * 48
    SIGNER_SERVICE_ACCOUNT_EMAIL_ENV_VAR: ClassVar[str] = 'SIGNER_SERVICE_ACCOUNT_EMAIL'

    # Instance attributes as Pydantic fields
    genai_client: Any = None
    gcs_bucket_name: str = ""
    credentials: Any = None
    project_id: str = ""
    storage_client: Any = None
    signer_service_account_email: str = ""

    def __init__(self, **data):
        # Set default values for BaseAgent required fields
        data.setdefault('name', 'VideoGenerationAgent')
        data.setdefault('description', 'An agent that generates video from text prompts using VEO')
        
        super().__init__(**data)
        logger.info('Initializing VideoGenerationAgent...')
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize Google GenAI client: prefer ADC (Cloud) for VEO, fallback to API key
        self.genai_client = None
        try:
            # Attempt to use Application Default Credentials (Cloud, billed)
            _adc_creds, _adc_project = google.auth.default(
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            self.genai_client = genai.Client()
            logger.info('Google GenAI client initialized with ADC.')
        except Exception as adc_err:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.error(
                    'Failed to initialize GenAI client with ADC and no GOOGLE_API_KEY set.'
                )
                raise
            try:
                self.genai_client = genai.Client(api_key=api_key)
                logger.info('Google GenAI client initialized with API key.')
            except Exception as e:
                logger.error(f'Failed to initialize Google GenAI client with API key: {e}')
                self.genai_client = None
                raise

        self.gcs_bucket_name = os.getenv(self.GCS_BUCKET_NAME_ENV_VAR)
        if not self.gcs_bucket_name:
            logger.error(
                f'{self.GCS_BUCKET_NAME_ENV_VAR} environment variable not set. '
                'Video upload to GCS will not be possible.'
            )
            raise
        if not storage:
            logger.error(
                'google-cloud-storage library not found, but GCS bucket is set. '
                'Video upload to GCS will fail. Please install google-cloud-storage.'
            )
            raise
        try:
            self.credentials, self.project_id = google.auth.default(
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            logger.info('Successfully obtained ADC for GCS.')
            self.storage_client = storage.Client(
                credentials=self.credentials, project=self.project_id
            )
            logger.info('Google Cloud Storage client initialized.')
        except google.auth.exceptions.DefaultCredentialsError:
            logger.error(
                'Could not get Application Default Credentials for GCS. '
                "Please run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS."
            )
            raise
        except Exception as e:
            logger.error(
                f'Failed to initialize Google Cloud Storage client: {e}'
            )
            raise

        sa_email_from_env = os.getenv(self.SIGNER_SERVICE_ACCOUNT_EMAIL_ENV_VAR)
        self.signer_service_account_email = (
            sa_email_from_env.strip('\'"') if sa_email_from_env else None
        )
        if self.signer_service_account_email:
            logger.info(
                f"Will use service account '{self.signer_service_account_email}' for signing GCS URLs."
            )
        else:
            logger.info(
                'No SIGNER_SERVICE_ACCOUNT_EMAIL set. Will use ambient gcloud credentials for signing GCS URLs.'
            )

        logger.info('VideoGenerationAgent initialized.')

    async def _generate_signed_url(
        self, blob_name: str, bucket_name: str, expiration_seconds: int
    ) -> str:
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        try:
            signed_url = blob.generate_signed_url(
                version='v4',
                expiration=expiration_seconds,
                method='GET',
                service_account_email=self.signer_service_account_email,  # None if not set, uses ambient creds
            )
            logger.info(
                f'Successfully generated signed URL for gs://{bucket_name}/{blob_name}'
            )
            return signed_url
        except Exception as e:
            logger.error(
                f'Error generating signed URL for gs://{bucket_name}/{blob_name}: {e}. '
                f"Check permissions (e.g., 'Service Account Token Creator' if using impersonation). "
                f'Falling back to GCS URI.'
            )
            return f'gs://{bucket_name}/{blob_name}'

    async def stream(
        self, prompt: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """Handles streaming requests for video generation.
        Yields progress updates and the final video URL.
        `session_id` is the A2A Task ID, used here for logging and unique naming.
        """
        logger.info(
            f"VideoGenerationAgent stream started for session_id: {session_id}, prompt: '{prompt}'"
        )

        yield {
            'is_task_complete': False,
            'updates': f"Received prompt: '{prompt}'. Starting VEO video generation.",
            'progress_percent': 0,
        }

        start_time = time.monotonic()
        operation_kicked_off = False
        veo_operation_name_for_reporting = 'N/A'
        try:
            logger.info(
                f'[{session_id}] Calling VEO with model: {self.VEO_MODEL_NAME}'
            )
            # Prepare a suggested output subpath (for potential future uploads if needed)
            veo_output_subpath = f'{session_id}/veo_direct_output/{uuid.uuid4()}'

            veo_operation = await asyncio.to_thread(
                self.genai_client.models.generate_videos,
                model=self.VEO_MODEL_NAME,
                prompt=prompt,
                config=genai_types.GenerateVideosConfig(
                    person_generation=self.VEO_DEFAULT_PERSON_GENERATION,
                    aspect_ratio=self.VEO_DEFAULT_ASPECT_RATIO,
                ),
            )
            if hasattr(veo_operation, 'name') and veo_operation.name:
                veo_operation_name_for_reporting = veo_operation.name
            else:
                logger.warning(
                    f"[{session_id}] Initial VEO operation object lacks a 'name' attribute or it's empty. Object: {str(veo_operation)[:200]}"
                )

            operation_kicked_off = True
            logger.info(
                f'[{session_id}] VEO operation started: {veo_operation_name_for_reporting}'
            )
            yield {
                'is_task_complete': False,
                'updates': f"VEO operation '{veo_operation_name_for_reporting}' started. Polling for completion...",
                'progress_percent': 5,  # Small initial progress
            }

            while True:
                if not hasattr(veo_operation, 'done'):
                    error_msg = f"[{session_id}] VEO operation variable is not a valid operation object before 'done' check. Type: {type(veo_operation)}, Value: {str(veo_operation)[:200]}"
                    logger.error(error_msg)
                    raise TypeError(error_msg)

                if veo_operation.done:
                    break  # Exit polling loop

                await asyncio.sleep(self.VEO_POLLING_INTERVAL_SECONDS)

                polled_data = await asyncio.to_thread(
                    self.genai_client.operations.get, veo_operation
                )

                if hasattr(polled_data, 'done') and hasattr(
                    polled_data, 'name'
                ):
                    veo_operation = polled_data
                    if veo_operation.name:
                        veo_operation_name_for_reporting = veo_operation.name
                else:
                    error_msg = f"[{session_id}] VEO polling for '{veo_operation_name_for_reporting}' returned unexpected data type: {type(polled_data)}. Value: {str(polled_data)[:200]}"
                    logger.error(error_msg)
                    # Yield an error and exit stream, as we can't continue polling
                    yield {
                        'is_task_complete': True,
                        'content': error_msg,
                        'final_message_text': 'Video generation polling encountered an API issue.',
                        'progress_percent': 100,
                    }
                    return

                elapsed_time = time.monotonic() - start_time
                simulated_progress = min(
                    int(
                        (
                            elapsed_time
                            / self.VEO_SIMULATED_TOTAL_GENERATION_TIME_SECONDS
                        )
                        * 100
                    ),
                    99,
                )
                current_progress = max(5, simulated_progress)
                yield {
                    'is_task_complete': False,
                    'updates': f'Video generation in progress (Operation: {veo_operation_name_for_reporting}). Simulated progress: {current_progress}%',
                    'progress_percent': current_progress,
                }

            logger.info(
                f'[{session_id}] VEO operation {veo_operation.name} is_done: {veo_operation.done}'
            )

            if veo_operation.error:
                error_message_detail = getattr(
                    veo_operation.error, 'message', str(veo_operation.error)
                )
                error_message = (
                    f'VEO video generation failed: {error_message_detail}'
                )
                logger.error(
                    f'[{session_id}] {error_message} (Raw error: {veo_operation.error})'
                )
                yield {
                    'is_task_complete': True,
                    'content': error_message,
                    'is_error': True,
                    'final_message_text': error_message,
                    'progress_percent': 100,
                }
                return

            logger.debug(
                f'[{session_id}] VEO operation completed. Response: {str(veo_operation.response)[:500]}...'
            )  # Log truncated response

            if (
                veo_operation.response
                and veo_operation.response.generated_videos
            ):
                # Assuming we use the first generated video
                generated_video_info = veo_operation.response.generated_videos[
                    0
                ]
                video_obj = (
                    generated_video_info.video
                )  # Assumption: video_obj is always present

                mime_type = 'video/mp4'
                mime_type = getattr(video_obj, 'mime_type', None) or mime_type
                veo_provided_uri = getattr(video_obj, 'uri', None)

                logger.info(
                    f'[{session_id}] Video object received. URI: {veo_provided_uri}, MimeType: {mime_type}'
                )

                if not veo_provided_uri:
                    err_message = 'VEO response video_obj has no URI. Cannot provide a link to the generated video.'
                    logger.error(f'[{session_id}] {err_message}')
                    yield {
                        'is_task_complete': True,
                        'content': err_message,
                        'is_error': True,
                        'final_message_text': err_message,
                        'progress_percent': 100,
                    }
                    return

                # If we got a GCS URI, sign it; otherwise, use the URI as-is (likely https)
                if veo_provided_uri.startswith('gs://'):
                    # Parse the GCS URI provided by VEO
                    try:
                        parsed_uri = urlparse(veo_provided_uri)
                        veo_bucket_name = parsed_uri.netloc
                        veo_blob_name = parsed_uri.path.lstrip('/')
                        logger.info(
                            f'[{session_id}] Parsed VEO GCS URI. Bucket: {veo_bucket_name}, Blob: {veo_blob_name}'
                        )
                    except Exception as e:
                        logger.error(
                            f"[{session_id}] Failed to parse VEO GCS URI '{veo_provided_uri}': {e}"
                        )
                        yield {
                            'is_task_complete': True,
                            'content': f'Failed to parse VEO GCS URI: {veo_provided_uri}',
                            'is_error': True,
                            'final_message_text': 'Video processing error.',
                            'progress_percent': 100,
                        }
                        return

                    if veo_bucket_name and veo_blob_name:
                        signed_gcs_url = await self._generate_signed_url(
                            veo_blob_name,
                            veo_bucket_name,
                            self.SIGNED_URL_EXPIRATION_SECONDS,
                        )
                        video_filename_for_artifact = veo_provided_uri.split('/')[-1]
                        artifact_description = (
                            f"Generated video for prompt: '{prompt}'. Original GCS location: {veo_provided_uri}"
                        )
                        completion_message = (
                            f'Video generation successful. Access video at link (expires): {signed_gcs_url}. '
                            f'Original GCS location: {veo_provided_uri}'
                        )
                        if signed_gcs_url == veo_provided_uri:
                            completion_message = (
                                f'Video generation successful. Video stored at GCS: {veo_provided_uri}. '
                                'A signed URL could not be generated.'
                            )
                            logger.warning(
                                f'[{session_id}] Signed URL generation might have failed or was not applicable, using GCS URI: {veo_provided_uri}'
                            )
                    else:
                        err_message = (
                            "VEO generation completed, but failed to parse bucket/blob from VEO's GCS URI for signing."
                        )
                        logger.error(f'[{session_id}] {err_message} (URI was {veo_provided_uri})')
                        yield {
                            'is_task_complete': True,
                            'content': err_message,
                            'is_error': True,
                            'final_message_text': err_message,
                            'progress_percent': 100,
                        }
                        return
                else:
                    # Use the provided HTTPS (or other) URI directly
                    signed_gcs_url = veo_provided_uri
                    try:
                        parsed = urlparse(veo_provided_uri)
                        last_segment = parsed.path.split('/')[-1]
                    except Exception:
                        last_segment = ''
                    video_filename_for_artifact = last_segment or f'video_{session_id}.mp4'
                    artifact_description = f"Generated video for prompt: '{prompt}'. Returned URI: {veo_provided_uri}"
                    completion_message = f'Video generation successful. Access video at: {veo_provided_uri}'

                logger.info(
                    f'[{session_id}] Yielding final success. URL: {signed_gcs_url}, Artifact Name: {video_filename_for_artifact}'
                )
                yield {
                    'is_task_complete': True,
                    'file_part_data': {
                        'uri': signed_gcs_url,
                        'mime_type': mime_type,
                    },
                    'artifact_name': video_filename_for_artifact,
                    'artifact_description': artifact_description,
                    'final_message_text': completion_message,
                    'progress_percent': 100,
                }

            elif (
                hasattr(veo_operation.response, 'rai_media_filtered_count')
                and veo_operation.response.rai_media_filtered_count > 0
            ):
                reasons = getattr(
                    veo_operation.response,
                    'rai_media_filtered_reasons',
                    ['Unknown safety filter.'],
                )
                message = f'Video generation was blocked by safety filters. Reasons: {", ".join(str(r) for r in reasons)}'
                logger.warning(f'[{session_id}] {message}')
                yield {
                    'is_task_complete': True,
                    'content': message,
                    'is_error': True,
                    'final_message_text': message,
                    'progress_percent': 100,
                }
            else:
                message = 'VEO generation completed, but no video was returned in the response and no explicit safety filter indicated.'
                logger.error(
                    f'[{session_id}] {message} Full response: {str(veo_operation.response)[:500]}'
                )
                yield {
                    'is_task_complete': True,
                    'content': message,
                    'is_error': True,
                    'final_message_text': message,
                    'progress_percent': 100,
                }

        except Exception as e:
            # Provide a clearer message for common precondition/billing errors
            err_text = str(e)
            billing_hint = ''
            if 'FAILED_PRECONDITION' in err_text or 'billing' in err_text.lower():
                billing_hint = (
                    " This model requires a Google Cloud project with billing enabled. "
                    "Ensure your current credentials point to a billed project and that Generative AI access is enabled."
                )
            error_context_msg = (
                f'VEO operation name: {veo_operation_name_for_reporting}'
                if operation_kicked_off
                else 'VEO operation not started.'
            )
            error_message = (
                f'An error occurred during video generation stream for session_id {session_id}: {e}. '
                f'Context: {error_context_msg}.{billing_hint}'
            )
            logger.exception(error_message)  # Log with traceback
            yield {
                'is_task_complete': True,
                'content': error_message,
                'is_error': True,
                'final_message_text': f'An unexpected error occurred: {e}{billing_hint}',
                'progress_percent': 100,
            }

    async def run_async(self, parent_context: InvocationContext) -> AsyncGenerator[Event, None]:
        """ADK-required method to run the agent asynchronously.
        
        This method adapts the existing stream method to work with ADK's Event system.
        """
        # Extract the prompt from InvocationContext.user_content
        import logging
        logging.info("parent_context dir: %s", dir(parent_context))
        prompt = ""
        session_id = str(uuid.uuid4())

        def _extract_text_from_content(content_obj: Any) -> str:
            try:
                parts = getattr(content_obj, 'parts', None)
                if parts:
                    texts: list[str] = []
                    for p in parts:
                        t = getattr(p, 'text', None)
                        if t:
                            texts.append(str(t))
                    return " ".join(texts).strip()
            except Exception as e:
                logger.debug("Failed to extract text from content: %s", e)
            return ""

        try:
            user_content = getattr(parent_context, 'user_content', None)
            logging.info("InvocationContext.user_content type: %s", type(user_content))
            if user_content:
                if isinstance(user_content, (list, tuple)):
                    prompt = " ".join(
                        filter(None, (_extract_text_from_content(c) for c in user_content))
                    ).strip()
                else:
                    prompt = _extract_text_from_content(user_content)
        except Exception as e:
            logger.debug("Error while extracting prompt from user_content: %s", e)
            
        # Ensure we have a fallback prompt if not provided by the UI/context
        if not prompt:
            prompt = "Generate a video"

        # Convert the stream output to ADK Events following ADK guidelines
        async for update in self.stream(prompt, session_id):
            progress = update.get('progress_percent')
            # Prefer a human-readable status line for the UI
            text = (
                update.get('updates')
                or update.get('final_message_text')
                or update.get('content')
            )

            event_kwargs: dict[str, Any] = {
                'author': self.name,
            }
            if progress is not None:
                try:
                    event_kwargs['actions'] = EventActions(
                        state_delta={'progress_percent': int(progress)}
                    )
                except Exception:
                    # Be defensive; if conversion fails, skip state delta
                    logger.warning(
                        "Invalid progress value encountered: %s", progress
                    )
            if text:
                event_kwargs['content'] = genai_types.Content(
                    parts=[genai_types.Part(text=str(text))]
                )

            # Emit a single well-formed Event per update
            yield Event(**event_kwargs)

