"""Unit tests for pii_redact.ner.entity_filters.

Pure Python, no ML imports. All tests run in the fast tier.
"""

from __future__ import annotations

import pytest
from pii_redact.ner.entity_filters import is_valid_org, is_valid_person

pytestmark = pytest.mark.fast


class TestIsValidOrg:
    # --- True positives: real company names that must pass ---

    def test_company_with_inc_suffix(self):
        assert is_valid_org("GreenTech Inc.") is True

    def test_company_with_llc_suffix(self):
        assert is_valid_org("Blue Ridge LLC") is True

    def test_company_with_ltd_suffix(self):
        assert is_valid_org("Eurobank AG") is True

    def test_hyphenated_company_name(self):
        assert is_valid_org("Cameron-Mcknight") is True

    def test_bank_name_in_financial_context(self):
        ctx = "The borrower has an account at ABC Bank for over 10 years."
        assert is_valid_org("ABC Bank", ctx) is True

    def test_employer_in_employment_context(self):
        ctx = "Patient is employed at Riverside Medical Group since 2019."
        assert is_valid_org("Riverside Medical Group", ctx) is True

    def test_multiword_with_legal_suffix(self):
        assert is_valid_org("ABC Research and Development Inc.") is True

    # --- False positives: role words and artifacts that must fail ---

    def test_generic_client_label(self):
        assert is_valid_org("Client") is False

    def test_generic_vendor_label(self):
        assert is_valid_org("Vendor") is False

    def test_service_provider_label(self):
        assert is_valid_org("Service Provider") is False

    def test_customer_service_label(self):
        assert is_valid_org("Customer Service") is False

    def test_all_caps_acronym_short(self):
        assert is_valid_org("CMT") is False

    def test_all_caps_acronym_ai(self):
        assert is_valid_org("AI") is False

    def test_all_caps_acronym_pii(self):
        assert is_valid_org("PII") is False

    def test_all_caps_acronym_hmrc(self):
        assert is_valid_org("HMRC") is False

    def test_xml_schema_artifact(self):
        assert is_valid_org('schemaLocation="http://www.fpml.org/FpML-5"') is False

    def test_lowercase_starting_span(self):
        assert is_valid_org("the Loan Amount") is False

    def test_empty_string(self):
        assert is_valid_org("") is False

    def test_whitespace_only(self):
        assert is_valid_org("   ") is False

    def test_forward_slash_artifact(self):
        assert is_valid_org("CUST/SMITH/") is False

    def test_at_sign_artifact(self):
        assert is_valid_org("admin@domain.com") is False

    def test_very_long_phrase_no_suffix(self):
        assert is_valid_org("This Supply Chain Management Agreement for the period") is False

    def test_short_ambiguous_no_context_passes_conservatively(self):
        # When context_window is absent we can't judge → keep (conservative, avoids FN in production)
        assert is_valid_org("Adobe", "") is True

    def test_short_ambiguous_with_nonfinancial_context_filtered(self):
        # "Adobe" with irrelevant context and no financial/legal signal = filtered
        assert is_valid_org("Adobe", "Please contact us at the above address.") is False

    def test_short_ambiguous_with_financial_context_passes(self):
        # "Adobe" near "lender" = allowed
        assert is_valid_org("Adobe", "The lender Adobe has provided financing.") is True


class TestIsValidPerson:
    # --- True positives ---

    def test_full_name(self):
        assert is_valid_person("Jane M. Doe") is True

    def test_first_last_name(self):
        assert is_valid_person("Robert Doe") is True

    def test_hyphenated_name(self):
        assert is_valid_person("Mary-Anne Smith") is True

    # --- False positives: field labels misclassified as PERSON ---

    def test_email_label(self):
        assert is_valid_person("Email") is False

    def test_phone_label(self):
        assert is_valid_person("Phone") is False

    def test_address_label(self):
        assert is_valid_person("Address") is False

    def test_vendor_label(self):
        assert is_valid_person("Vendor") is False

    def test_lowercase_starting(self):
        assert is_valid_person("los términos") is False

    def test_empty_string(self):
        assert is_valid_person("") is False
