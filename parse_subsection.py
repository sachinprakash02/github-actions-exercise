import re

class SectionParser:
    """
    Very reliable section/subsection detection for PDFs extracted using PyMuPDF.
    Uses THREE signals:
        - numbering pattern (biggest signal)
        - font size relative to median
        - vertical spacing (gap before block)
    """

    HEADING_PATTERN = re.compile(r"^\d+(\.\d+)*\s+")

    def parse(self, blocks):
        out = []
        current_section = None
        current_subsection = None

        # Precompute median font size to classify normal paragraphs
        font_sizes = [b["avg_font"] for b in blocks]
        median_font = self._median(font_sizes) or 10

        # compute vertical spacing
        prev_page = -1
        prev_y1 = -1

        for blk in blocks:
            text = blk["text"]
            page = blk["page"]
            font = blk["avg_font"]
            x0, y0, x1, y1 = blk["bbox"]

            # reset per page
            if page != prev_page:
                prev_y1 = -1
                prev_page = page

            vertical_gap = (y0 - prev_y1) if prev_y1 >= 0 else 0
            prev_y1 = y1

            # ---------------------------------------------------------
            # Signal 1: numbering pattern → strongest indicator of heading
            # ---------------------------------------------------------
            is_numbered_heading = bool(self.HEADING_PATTERN.match(text))

            # ---------------------------------------------------------
            # Signal 2: large font (compared to median)
            # ---------------------------------------------------------
            is_large_font = font >= median_font * 1.35

            # ---------------------------------------------------------
            # Signal 3: vertical spacing
            # headings usually have big gap before them
            # ---------------------------------------------------------
            is_vertical_heading = vertical_gap > 20  # tuning threshold

            # Final heading decision
            is_heading = is_numbered_heading or is_large_font or is_vertical_heading

            if is_heading:
                # Determine section vs subsection
                if is_numbered_heading:
                    # Depth based on number of dots
                    depth = text.split()[0].count(".")
                    if depth == 0:
                        current_section = text
                        current_subsection = None
                    elif depth == 1:
                        current_subsection = text
                    else:
                        # deeper levels treated as subsections
                        current_subsection = text

                out.append({
                    "text": text,
                    "page": page,
                    "section": current_section,
                    "subsection": current_subsection,
                    "block_type": "heading"
                })
            else:
                out.append({
                    "text": text,
                    "page": page,
                    "section": current_section,
                    "subsection": current_subsection,
                    "block_type": "paragraph"
                })

        return out


    def _median(self, values):
        if not values:
            return None
        values = sorted(values)
        mid = len(values) // 2
        if len(values) % 2 == 0:
            return (values[mid-1] + values[mid]) / 2
        return values[mid]
