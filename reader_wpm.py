import streamlit as st
import time
import re

st.set_page_config(page_title="RSVP Speed Reader", layout="centered")

SAMPLE_TEXTS = {
    # wikipedia
    #wikipedia
    "Hlapci - Cankar": (
        "Komar je moral tudi poklekniti pred učiteljem Hvastjo, v znak pokore, ki mu jo je učitelj naložil in hvali se s tem," 
        "da je ponižnost čednost kristjana. Medtem v zbornico prideta tudi Lojzka in Geni, ki sta se že spreobrnili in"
        "onidve se hvalita z rožnim vencem."
        " Medtem pride župnik, ki ga vsi hitijo pozdravit. Prišel je, da bi jih povabil na večerjo." 
        "Z zamudo vstopi še Minka, ki se opraviči s tem, da je bila pri spovedi. Župnik je, kot oseba hladen in trd človek," 
        "ki učitelje ošteva zaradi njihove preteklosti in jih poziva k ponižnosti."
    ),
}

st.markdown("""
<style>
    .rsvp-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 120px;
        background-color: #f8f8f6;
        border-radius: 12px;
        border: 1px solid #e0e0dc;
        margin: 1rem 0;
    }
    .rsvp-word {
        font-size: 2.6rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        font-family: 'Georgia', serif;
    }
    .pivot { color: #D85A30; }
    .side  { color: #888; }
    .stat-box {
        background: #f2f2ee;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
    }
    .stat-label { font-size: 0.75rem; color: #888; margin-bottom: 2px; }
    .stat-value { font-size: 1.1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def get_pivot_index(word: str) -> int:
    clean = re.sub(r"[^a-zA-Z0-9]", "", word)
    n = len(clean)
    if n <= 1: return 0
    if n <= 5: return 1
    if n <= 9: return 2
    return 3


def render_word(word: str) -> str:
    if not word:
        return '<div class="rsvp-container"><span class="rsvp-word"><span class="pivot">·</span></span></div>'
    pi = get_pivot_index(word)
    before = word[:pi]
    pivot  = word[pi] if pi < len(word) else ""
    after  = word[pi+1:] if pi + 1 < len(word) else ""
    html = '<div class="rsvp-container"><span class="rsvp-word">'
    if before:
        html += f'<span class="side">{before}</span>'
    html += f'<span class="pivot">{pivot}</span>'
    if after:
        html += f'<span class="side">{after}</span>'
    html += "</span></div>"
    return html


def tokenize(text: str) -> list[str]:
    return [w for w in text.strip().split() if w]


if "playing"    not in st.session_state: st.session_state.playing    = False
if "word_index" not in st.session_state: st.session_state.word_index = 0
if "words"      not in st.session_state: st.session_state.words      = []
if "start_time" not in st.session_state: st.session_state.start_time = None
if "elapsed"    not in st.session_state: st.session_state.elapsed    = 0.0


st.title("RSVP Speed Reader")
st.caption("Rapid Serial Visual Presentation — words flash, so your eyes dont need tom ove. To stop - control+C")

with st.sidebar:
    st.header("Settings")
    wpm = st.slider("Speed (words per minute)", min_value=100, max_value=700, value=300, step=25)
    st.markdown("---")
    st.subheader("Sample texts")
    for label, body in SAMPLE_TEXTS.items():
        if st.button(label, use_container_width=True):
            st.session_state.playing    = False
            st.session_state.word_index = 0
            st.session_state.words      = tokenize(body)
            st.session_state.elapsed    = 0.0
            st.session_state["custom_text"] = body


custom_text = st.text_area(
    "your text - opzy it here",
    value=st.session_state.get("custom_text", SAMPLE_TEXTS["Hlapci - Cankar"]),
    height=120,
    key="custom_text",
)

words = tokenize(custom_text)
if words != st.session_state.words:
    st.session_state.words      = words
    st.session_state.word_index = 0
    st.session_state.playing    = False
    st.session_state.elapsed    = 0.0

total = len(st.session_state.words)
idx   = st.session_state.word_index

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-box"><div class="stat-label">Words</div><div class="stat-value">{total}</div></div>', unsafe_allow_html=True)
with c2:
    est_min = round(total / wpm, 1) if total else 0
    st.markdown(f'<div class="stat-box"><div class="stat-label">Est. time</div><div class="stat-value">{est_min} min</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="stat-label">Position</div><div class="stat-value">{idx}/{total}</div></div>', unsafe_allow_html=True)
with c4:
    pct = round(100 * idx / total) if total else 0
    st.markdown(f'<div class="stat-box"><div class="stat-label">Progress</div><div class="stat-value">{pct}%</div></div>', unsafe_allow_html=True)

word_display = st.empty()

if total and idx < total:
    word_display.markdown(render_word(st.session_state.words[idx]), unsafe_allow_html=True)
elif idx >= total and total:
    word_display.markdown(render_word("-"), unsafe_allow_html=True)
else:
    word_display.markdown(render_word(""), unsafe_allow_html=True)

progress_bar = st.progress(pct / 100 if total else 0)

# copy/paste from net! :)
b1, b2, b3, b4 = st.columns([2, 1, 1, 1])
with b1:
    play_label = "⏸ Pause" if st.session_state.playing else ("▶ Restart" if idx >= total and total else "▶ Play")
    if st.button(play_label, use_container_width=True, type="primary"):
        if st.session_state.playing:
            st.session_state.playing = False
        else:
            if idx >= total:
                st.session_state.word_index = 0
            st.session_state.playing   = True
            st.session_state.start_time = time.time()
with b2:
    if st.button("⏮ Restart", use_container_width=True):
        st.session_state.playing    = False
        st.session_state.word_index = 0
        st.session_state.elapsed    = 0.0
        st.rerun()
with b3:
    if st.button("◀ Back", use_container_width=True):
        st.session_state.playing    = False
        st.session_state.word_index = max(0, st.session_state.word_index - 10)
        st.rerun()
with b4:
    if st.button("Forward ▶", use_container_width=True):
        st.session_state.playing    = False
        st.session_state.word_index = min(total - 1, st.session_state.word_index + 10)
        st.rerun()

if st.session_state.playing and st.session_state.words:
    delay = 60.0 / wpm
    i = st.session_state.word_index

    if i < total:
        word_display.markdown(render_word(st.session_state.words[i]), unsafe_allow_html=True)
        progress_bar.progress((i + 1) / total)
        st.session_state.word_index += 1
        time.sleep(delay)
        st.rerun()
    else:
        st.session_state.playing = False
        word_display.markdown(render_word("Done"), unsafe_allow_html=True)
        progress_bar.progress(1.0)
        st.rerun()