"""OpenTelemetry setup and configuration for Second Brain."""

import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

# Suppress noisy OTEL connection errors and console output
logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)  # Suppress all OTEL logs
logging.getLogger("logfire").setLevel(logging.ERROR)  # Suppress logfire console logs

# Configure logfire for enhanced tracing (optional, helps with pydantic-ai instrumentation)
# This runs at module import time to ensure instrumentation happens before pydantic_ai is imported
try:
    import logfire
    
    # Set OTEL endpoint for logfire (before configuring)
    if "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    
    # Configure logfire to send to OTEL-TUI (not logfire cloud)
    # This sets up logfire's own tracer provider
    # Disable console output - only send to OTEL endpoint
    logfire.configure(
        send_to_logfire=False,
        console=False,  # Disable console output
        service_name="second-brain",
        service_version="0.1.0"
    )
    
    # Instrument pydantic-ai for better tracing (must be before pydantic_ai is imported)
    logfire.instrument_pydantic_ai()
    
    # Instrument httpx for HTTP request tracing
    logfire.instrument_httpx(capture_all=True)
    
    LOGFIRE_ENABLED = True
    
except ImportError:
    # Logfire not installed, continue without it
    LOGFIRE_ENABLED = False
except Exception as e:
    # If logfire setup fails, continue without it (non-blocking)
    LOGFIRE_ENABLED = False
    logging.getLogger(__name__).warning(f"Logfire setup failed: {e}, continuing without it")


def setup_otel():
    """Initialize OpenTelemetry with OTLP exporter for OTEL Desktop."""
    
    # Check if OTEL is disabled
    if os.getenv("OTEL_DISABLED", "false").lower() == "true":
        return trace.get_tracer(__name__)
    
    # If logfire is enabled, it already set up the tracer provider
    # Just return a tracer from the existing provider
    if LOGFIRE_ENABLED:
        return trace.get_tracer(__name__)
    
    # Create resource with service name
    resource = Resource.create({
        "service.name": "second-brain",
        "service.version": "0.1.0",
    })
    
    # Set up tracer provider (only if logfire didn't do it)
    tracer_provider = TracerProvider(resource=resource)
    
    # Configure OTLP exporter for OTEL-TUI/Jaeger
    # OTEL-TUI listens on localhost:4318 (HTTP) or localhost:4317 (gRPC)
    # Use OTEL_EXPORTER_OTLP_ENDPOINT env var or default to OTEL-TUI
    otlp_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4318/v1/traces"  # Default endpoint for OTEL-TUI/Jaeger
    )
    
    try:
        # OTLPSpanExporter automatically appends /v1/traces, but if endpoint already has it, that's fine
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            headers={}  # Add any required headers here if needed
        )
        
        # Reduce batching delay for faster trace visibility (helps with OTEL-TUI)
        # BatchSpanProcessor batches spans, use shorter export delay for better UX
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_export_batch_size=512,
            export_timeout_millis=30000,
            schedule_delay_millis=500,  # Flush every 500ms instead of default 5s for faster visibility
        )
        
        # Add span processor with faster flushing
        tracer_provider.add_span_processor(span_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
    except Exception as e:
        # If OTEL setup fails, continue without it (non-blocking)
        print(f"⚠️  OTEL setup warning: {str(e)}")
        print("   Continuing without observability. Start OTEL Desktop to enable tracing.")
        # Create a no-op tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
    
    return trace.get_tracer(__name__)


def get_tracer(name: str = __name__):
    """Get a tracer instance."""
    return trace.get_tracer(name)

