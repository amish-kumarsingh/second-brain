"""Guardrails utility for PII detection and redaction."""

import os
import re
from typing import Optional
from second_brain.utils import get_tracer
from opentelemetry.trace import Status, StatusCode

tracer = get_tracer("second_brain.guardrails")

# Try to import guardrails-ai (lazy import to avoid compatibility issues)
GUARDRAILS_AVAILABLE = False
Guard = None  # type: ignore[assignment, misc]
DetectPII = None  # type: ignore[assignment, misc]
_guardrails_import_tried = False  # Track if we've already tried and failed
_message_printed = False  # Track if we've already printed the status message


def _try_import_guardrails():
    """Lazy import guardrails to avoid compatibility issues at module load time."""
    global GUARDRAILS_AVAILABLE, Guard, DetectPII, _guardrails_import_tried
    if GUARDRAILS_AVAILABLE:
        return True
    
    # If we've already tried and failed, don't try again
    if _guardrails_import_tried:
        return False
    
    try:
        from guardrails import Guard  # type: ignore[import-untyped]
        from guardrails.hub import DetectPII  # type: ignore[import-untyped]
        Guard = Guard
        DetectPII = DetectPII
        GUARDRAILS_AVAILABLE = True
        _guardrails_import_tried = True
        return True
    except (ImportError, AttributeError, Exception) as e:
        # Guardrails might have compatibility issues (e.g., with OpenAI version)
        GUARDRAILS_AVAILABLE = False
        _guardrails_import_tried = True
        return False


# Regex patterns for common PII types (fallback if guardrails unavailable)
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}


class PIIGuard:
    """Guardrails wrapper for PII detection and redaction."""
    
    def __init__(self, enable_guardrails: Optional[bool] = None):
        """
        Initialize PII guard.
        
        Args:
            enable_guardrails: If None, checks GUARDRAILS_ENABLED env var. Defaults to True if guardrails is available.
        """
        global _message_printed
        
        self.enabled = False
        self.guard = None
        
        # Enable by default (can be disabled via env var)
        if enable_guardrails is None:
            self.enabled = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
        else:
            self.enabled = enable_guardrails
        
        # Always enabled for PII protection (uses guardrails if available, regex as fallback)
        if self.enabled:
            # Try to use guardrails if available
            if _try_import_guardrails():
                try:
                    # Initialize guard with PII detection
                    self.guard = Guard().use(DetectPII(  # type: ignore[misc]
                        pii_entities=[
                            "EMAIL_ADDRESS",
                            "PHONE_NUMBER",
                            "CREDIT_CARD",
                            "SSN",
                            "US_PASSPORT",
                            "IP_ADDRESS",
                            "DATE_TIME",
                            "PERSON",
                            "LOCATION",
                            "NRP",  # Nationality/Religious/Political groups
                            "ORGANIZATION",
                            "MEDICAL_LICENSE",
                            "US_BANK_NUMBER",
                            "CRYPTO",
                            "US_DRIVER_LICENSE"
                        ],
                        threshold=0.5,
                        redact=True  # Enable redaction mode
                    ))
                    # Only print once when successfully enabled
                    if not _message_printed:
                        print("✅ Guardrails AI PII protection enabled (using AI-based detection)")
                        _message_printed = True
                except Exception as e:
                    # If guardrails fails to initialize, fall back to regex
                    if not _message_printed:
                        print(f"⚠️  Guardrails AI not available due to compatibility issues.")
                        print("   Using regex-based PII detection (email, phone, SSN, credit card, IP).")
                        _message_printed = True
                    self.guard = None
            else:
                # guardrails-ai import failed (compatibility issue or not installed)
                if not _message_printed:
                    print("✅ PII protection enabled (using regex-based detection)")
                    _message_printed = True
                self.guard = None
    
    def _sanitize_regex(self, text: str) -> str:
        """Fallback regex-based PII redaction."""
        sanitized = text
        for pii_type, pattern in PII_PATTERNS.items():
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        return sanitized

    def sanitize(self, text: str) -> str:
        """
        Sanitize text by detecting and redacting PII.
        
        Args:
            text: Input text that may contain PII
        
        Returns:
            Text with PII redacted (replaced with [REDACTED])
        """
        if not self.enabled or not text:
            return text
        
        with tracer.start_as_current_span("guardrails.sanitize") as span:
            span.set_attribute("input_length", len(text))
            try:
                # Try guardrails first if available
                if self.guard is not None:
                    assert self.guard is not None
                    result = self.guard.validate(text)
                    
                    # Get sanitized output
                    sanitized_text = result.validated_output if hasattr(result, 'validated_output') else text
                    
                    # Check for PII entities detected
                    pii_detected = False
                    if hasattr(result, 'validation_passed') and not result.validation_passed:
                        pii_detected = True
                        if hasattr(result, 'error') and hasattr(result.error, 'fail_results'):
                            pii_count = len(result.error.fail_results) if result.error.fail_results else 0
                            span.set_attribute("pii_entities_count", pii_count)
                    
                    span.set_attribute("pii_detected", pii_detected)
                    span.set_attribute("method", "guardrails")
                else:
                    # Fallback to regex-based detection
                    sanitized_text = self._sanitize_regex(text)
                    span.set_attribute("method", "regex")
                    # Simple check if text changed (likely PII was found)
                    span.set_attribute("pii_detected", sanitized_text != text)
                
                span.set_attribute("output_length", len(sanitized_text))
                span.set_status(Status(StatusCode.OK))
                return sanitized_text
                
            except Exception as e:
                # If sanitization fails, try regex fallback
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                try:
                    sanitized_text = self._sanitize_regex(text)
                    span.set_attribute("method", "regex_fallback")
                    return sanitized_text
                except Exception:
                    # Last resort: return original text
                    print(f"⚠️  PII sanitization failed: {e}. Returning original text.")
                    return text
    
    def detect_pii(self, text: str) -> dict:
        """
        Detect PII in text without redacting.
        
        Args:
            text: Input text to check for PII
        
        Returns:
            Dictionary with detection results
        """
        if not self.enabled or not text:
            return {"has_pii": False, "entities": []}
        
        with tracer.start_as_current_span("guardrails.detect_pii") as span:
            try:
                if self.guard is None:
                    return {"has_pii": False, "entities": []}
                # Use guard to detect PII (without redaction)
                # Type check: guard is not None at this point
                assert self.guard is not None
                result = self.guard.validate(text)
                
                has_pii = False
                entities = []
                
                if hasattr(result, 'validation_passed') and not result.validation_passed:
                    has_pii = True
                    if hasattr(result, 'error') and hasattr(result.error, 'fail_results'):
                        entities = result.error.fail_results
                
                span.set_attribute("has_pii", has_pii)
                span.set_attribute("entities_count", len(entities))
                span.set_status(Status(StatusCode.OK))
                
                return {
                    "has_pii": has_pii,
                    "entities": entities
                }
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                return {"has_pii": False, "entities": []}


# Global instance (initialized on first use)
_guard_instance: Optional[PIIGuard] = None


def get_guard() -> PIIGuard:
    """Get or create the global PII guard instance."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = PIIGuard()
    return _guard_instance


def sanitize_text(text: str) -> str:
    """Convenience function to sanitize text using the global guard."""
    return get_guard().sanitize(text)

