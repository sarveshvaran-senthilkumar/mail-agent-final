from bson import ObjectId
import src.database as database
from src.models import Document


def test_find_or_create_user_creates_then_reuses(mock_mongo_client):
    user1 = database.find_or_create_user("a@example.com")
    user2 = database.find_or_create_user("a@example.com")

    assert user1.id == user2.id
    assert database.users_collection().count_documents({}) == 1


def test_insert_document(mock_mongo_client):
    user = database.find_or_create_user("a@example.com")
    doc = Document(
        user_id=user.id,
        message_id="msg-1",
        filename="invoice.pdf",
        doc_type="invoice",
        folder="invoices",
        storage_path="/storage/invoices/invoice.pdf",
        inbox_path="/storage/inbox/invoice.pdf",
        llm_tag="qwen:0.92",
    )
    doc_id = database.insert_document(doc)
    assert database.documents_collection().find_one({"_id": ObjectId(doc_id)}) is not None
    assert database.documents_collection().count_documents({}) == 1



def test_is_processed_and_mark_processed(mock_mongo_client):
    assert database.is_processed("msg-1") is False
    database.mark_processed("msg-1", internal_date_ms=0, status="success")
    assert database.is_processed("msg-1") is True


def test_insert_resume_analysis(mock_mongo_client):
    from src.models import ResumeAnalysis, StackExperience
    analysis = ResumeAnalysis(
        message_id="msg-resume-10",
        filename="candidate.pdf",
        stacks=[
            StackExperience(stack="java-developer", years=5.0, experience_band="level-2", confidence=0.9, evidence="5 yrs Java"),
        ],
    )
    analysis_id = database.insert_resume_analysis(analysis)
    assert database.resume_analyses_collection().find_one({"_id": ObjectId(analysis_id)}) is not None
    assert database.resume_analyses_collection().count_documents({}) == 1

