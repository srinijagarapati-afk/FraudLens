import streamlit as st
import fitz
import re
import spacy
import os
import pytesseract
import io

from PIL import Image
from dotenv import load_dotenv
from dateutil import parser
from google import genai
from sentence_transformers import SentenceTransformer, util


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(
        "GEMINI_API_KEY was not found. "
        "Please check your .env file."
    )
    st.stop()

client = genai.Client(api_key=api_key)


# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


nlp = load_spacy_model()
embedding_model = load_embedding_model()


# ==================================================
# USER INTERFACE
# ==================================================

st.title("🔍 FraudLens")

st.write(
    "AI-Powered Fraud Document Intelligence Platform"
)

st.write(
    "Upload a PDF document to extract information, "
    "ask questions, and analyze potential fraud indicators."
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf"]
)


# ==================================================
# PROCESS DOCUMENT
# ==================================================

if uploaded_file is not None:

    st.success("File uploaded successfully!")

    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    text = ""
    pages = []


    # ==================================================
    # TEXT EXTRACTION + OCR
    # ==================================================

    with st.spinner(
        "Extracting text from document..."
    ):

        for page_number, page in enumerate(
            pdf_document,
            start=1
        ):

            # First try normal PDF text extraction
            page_text = page.get_text()

            # If very little text exists,
            # try OCR because the page may be scanned.
            if len(page_text.strip()) < 20:

                try:

                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2)
                    )

                    image = Image.open(
                        io.BytesIO(
                            pix.tobytes("png")
                        )
                    )

                    page_text = (
                        pytesseract.image_to_string(
                            image
                        )
                    )

                except Exception as error:

                    st.warning(
                        f"OCR failed on page "
                        f"{page_number}: {error}"
                    )

            text += page_text + "\n"

            pages.append({
                "page_number": page_number,
                "text": page_text
            })


    # ==================================================
    # EXTRACTED TEXT
    # ==================================================

    st.subheader("📄 Extracted Text")

    if text.strip():

        with st.expander(
            "View extracted document text"
        ):
            st.write(text)

    else:

        st.warning(
            "No text could be extracted "
            "from this document."
        )


    # ==================================================
    # DOCUMENT CHUNKING
    # ==================================================

    chunk_size = 1000

    chunks = []
    chunk_pages = []

    for page in pages:

        page_text = page["text"]
        page_number = page["page_number"]

        for i in range(
            0,
            len(page_text),
            chunk_size
        ):

            chunk = page_text[
                i:i + chunk_size
            ]

            if chunk.strip():

                chunks.append(chunk)

                chunk_pages.append(
                    page_number
                )


    st.subheader("🧩 Document Processing")

    st.write(
        "Total chunks created:",
        len(chunks)
    )


    # ==================================================
    # EMBEDDINGS + RAG
    # ==================================================

    if chunks:

        with st.spinner(
            "Creating document embeddings..."
        ):

            embeddings = (
                embedding_model.encode(
                    chunks,
                    convert_to_tensor=True
                )
            )

        st.success(
            f"{len(embeddings)} embeddings created."
        )


        # ==================================================
        # ASK FRAUDLENS
        # ==================================================

        st.subheader("💬 Ask FraudLens")

        question = st.text_input(
            "Ask a question about the document",
            placeholder=(
                "Example: What is the total amount "
                "mentioned in this document?"
            )
        )


        if question:

            question_embedding = (
                embedding_model.encode(
                    question,
                    convert_to_tensor=True
                )
            )

            scores = util.cos_sim(
                question_embedding,
                embeddings
            )[0]


            # Retrieve top 3 relevant chunks
            number_of_results = min(
                3,
                len(chunks)
            )

            top_results = scores.topk(
                number_of_results
            )

            top_indices = (
                top_results.indices.tolist()
            )

            best_chunks = [
                chunks[i]
                for i in top_indices
            ]

            best_chunk = "\n\n".join(
                best_chunks
            )


            # Source pages
            source_pages = [
                chunk_pages[i]
                for i in top_indices
            ]

            source_pages = sorted(
                set(source_pages)
            )


            with st.expander(
                "🔎 View Most Relevant Evidence"
            ):
                st.write(best_chunk)


            # ==================================================
            # GEMINI PROMPT
            # ==================================================

            prompt = f"""
You are FraudLens, an AI fraud document
intelligence assistant.

Answer the user's question using ONLY
the document evidence provided below.

Do not invent information.

If the answer is not available in the
evidence, say:

"I could not find that information in
the document."

Question:
{question}

Document Evidence:
{best_chunk}
"""


            try:

                with st.spinner(
                    "FraudLens is analyzing..."
                ):

                    response = (
                        client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=prompt
                        )
                    )


                st.subheader(
                    "🤖 FraudLens Answer"
                )

                st.write(
                    response.text
                )


                st.subheader(
                    "📚 Sources"
                )

                for page_number in source_pages:

                    st.write(
                        f"📄 Page {page_number}"
                    )


            except Exception as error:

                st.error(
                    "Gemini is temporarily unavailable. "
                    "Please try again."
                )

                with st.expander(
                    "Technical error details"
                ):
                    st.write(
                        str(error)
                    )


    else:

        st.warning(
            "No document chunks could be created."
        )


    # ==================================================
    # FRAUD ANALYSIS
    # ==================================================

    st.divider()

    st.subheader(
        "🔎 Fraud Analysis"
    )


    # ==================================================
    # AMOUNT EXTRACTION
    # ==================================================

    amounts = re.findall(
        r'\$[\d,]+(?:\.\d{1,2})?',
        text
    )

    amounts = list(
        dict.fromkeys(amounts)
    )

    st.write(
        "💰 Detected Amounts:"
    )

    if amounts:
        st.write(amounts)
    else:
        st.write(
            "No dollar amounts detected."
        )


    # ==================================================
    # DATE EXTRACTION
    # ==================================================

    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',
        r'\b[A-Za-z]+ \d{1,2}, \d{4}\b',
        r'\b\d{1,2} [A-Za-z]+ \d{4}\b'
    ]

    dates = []

    for pattern in date_patterns:

        found_dates = re.findall(
            pattern,
            text
        )

        dates.extend(
            found_dates
        )

    dates = list(
        dict.fromkeys(dates)
    )


    st.write(
        "📅 Detected Dates:"
    )

    if dates:
        st.write(dates)
    else:
        st.write(
            "No dates detected."
        )


    # ==================================================
    # DATE NORMALIZATION
    # ==================================================

    normalized_dates = []

    for date in dates:

        try:

            parsed_date = parser.parse(
                date
            )

            normalized_dates.append({
                "Original": date,
                "Normalized": (
                    parsed_date.strftime(
                        "%Y-%m-%d"
                    )
                )
            })

        except (ValueError, TypeError):
            pass


    st.write(
        "📅 Normalized Dates:"
    )

    if normalized_dates:
        st.write(normalized_dates)
    else:
        st.write(
            "No dates available for normalization."
        )


    # ==================================================
    # LOCATION EXTRACTION
    # ==================================================

    known_locations = [
        "Dallas",
        "New York",
        "London",
        "Mumbai",
        "Singapore",
        "Toronto"
    ]

    locations = []

    for location in known_locations:

        if location.lower() in text.lower():

            locations.append(
                location
            )


    st.write(
        "📍 Detected Locations:"
    )

    if locations:
        st.write(locations)
    else:
        st.write(
            "No known locations detected."
        )


    # ==================================================
    # PEOPLE + ORGANIZATIONS
    # ==================================================

    people = []
    organizations = []

    if text.strip():

        with st.spinner(
            "Analyzing people and organizations..."
        ):

            doc = nlp(
                text[:1000000]
            )


        for entity in doc.ents:

            if entity.label_ == "PERSON":

                people.append(
                    entity.text.strip()
                )

            elif entity.label_ == "ORG":

                organizations.append(
                    entity.text.strip()
                )


        # Remove duplicates
        people = list(
            dict.fromkeys(people)
        )

        organizations = list(
            dict.fromkeys(organizations)
        )


    st.write(
        "👤 Detected People:"
    )

    if people:
        st.write(people)
    else:
        st.write(
            "No people detected."
        )


    st.write(
        "🏢 Detected Organizations:"
    )

    if organizations:
        st.write(organizations)
    else:
        st.write(
            "No organizations detected."
        )


    # ==================================================
    # DOCUMENT ANALYSIS SUMMARY
    # ==================================================

    st.divider()

    st.subheader(
        "📊 Document Analysis Summary"
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Pages",
            len(pages)
        )


    with col2:

        st.metric(
            "Amounts",
            len(amounts)
        )


    with col3:

        st.metric(
            "Dates",
            len(dates)
        )


    with col4:

        st.metric(
            "Locations",
            len(locations)
        )