import os


TEST_API_KEY = "test-user-api-key-0000000000000001"
TEST_ADMIN_API_KEY = "test-admin-api-key-00000000000001"
TEST_ORIGIN = "http://testserver"

os.environ.setdefault("JANUS_API_KEY", TEST_API_KEY)
os.environ.setdefault("JANUS_ADMIN_API_KEY", TEST_ADMIN_API_KEY)
os.environ.setdefault("JANUS_TRUSTED_ORIGINS", TEST_ORIGIN)
