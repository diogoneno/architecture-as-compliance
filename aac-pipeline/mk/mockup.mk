
MOCKUP_TOOLS ?= build/mockup-tools
MOCKUP_WIDTH ?= 1200
MOCKUP_PLAYWRIGHT := playwright==1.64.0
MOCKUP_UVRUN = env -u UV_PROJECT_ENVIRONMENT -u VIRTUAL_ENV $(UV) run --no-project --python 3.11 --with $(MOCKUP_PLAYWRIGHT)

.PHONY: mockup-screenshot

mockup-screenshot:
	@cd $(CURDIR); \
	if [ -z "$(HTML)" ] || [ -z "$(OUT)" ]; then echo "usage: make mockup-screenshot HTML=<page.html> OUT=<file.png>"; exit 2; fi; \
	mkdir -p $(MOCKUP_TOOLS); log=$(MOCKUP_TOOLS)/last-run.log; : > $$log; \
	if ! $(MOCKUP_UVRUN) python -m playwright install chromium >> $$log 2>&1; then \
 echo "mockup-screenshot: unavailable (playwright 1.64.0 or its chromium could not be fetched; see $$log)"; exit 1; fi; \
	line=$$($(MOCKUP_UVRUN) python scripts/screenshot_mockup.py "$(HTML)" "$(OUT)" --width $(MOCKUP_WIDTH) 2>>$$log | tail -1); \
	case "$$line" in \
 "mockup-screenshot: ok "*) echo "$$line"; exit 0 ;; \
 "mockup-screenshot: missing-libraries "*) echo "$$line" >> $$log; echo "mockup-screenshot: retry with extracted libnss3/libnspr4 ($$line)" ;; \
 "") echo "mockup-screenshot: unavailable (screenshot script printed nothing; see $$log)"; exit 1 ;; \
 *) echo "$$line"; exit 1 ;; \
	esac; \
	mkdir -p $(MOCKUP_TOOLS)/debs $(MOCKUP_TOOLS)/root; \
	if ! ( cd $(MOCKUP_TOOLS)/debs && apt-get download libnss3 libnspr4 ) >> $$log 2>&1; then \
 echo "mockup-screenshot: unavailable (browser lacks shared libraries and apt-get download libnss3 libnspr4 failed; see $$log)"; exit 1; fi; \
	for deb in $(MOCKUP_TOOLS)/debs/*.deb; do dpkg-deb -x "$$deb" $(MOCKUP_TOOLS)/root || { echo "mockup-screenshot: unavailable (dpkg-deb -x $$deb failed)"; exit 1; }; done; \
	libdir=$(abspath $(MOCKUP_TOOLS))/root/usr/lib/x86_64-linux-gnu; \
	line=$$(LD_LIBRARY_PATH="$$libdir:$$libdir/nss$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}" $(MOCKUP_UVRUN) python scripts/screenshot_mockup.py "$(HTML)" "$(OUT)" --width $(MOCKUP_WIDTH) 2>>$$log | tail -1); \
	case "$$line" in \
 "mockup-screenshot: ok "*) echo "$$line"; exit 0 ;; \
 "mockup-screenshot: missing-libraries ("*) echo "mockup-screenshot: unavailable (after the libnss3/libnspr4 retry: $${line#mockup-screenshot: missing-libraries (}"; exit 1 ;; \
 "") echo "mockup-screenshot: unavailable (screenshot script printed nothing on the retry; see $$log)"; exit 1 ;; \
 *) echo "$$line"; exit 1 ;; \
	esac
