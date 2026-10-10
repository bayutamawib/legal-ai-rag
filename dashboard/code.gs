const SHEET_NAME = "evaluation_stream";

const HEADERS = [
  "run_id",
  "sync_status",
  "case_id",
  "query",
  "query_type",
  "severity",
  "expected_answer",
  "expected_references",
  "answer",
  "source_count",
  "latency_seconds",
  "retrieved_articles",
  "retrieved_document_ids",
  "top_rerank_score",
  "expected_reference_count",
  "matched_reference_count",
  "retrieved_reference_count",
  "reference_recall",
  "reference_precision",
  "expected_references_flat",
  "matched_references",
  "retrieved_references",
  "abstention",
  "unexpected_abstention",
  "faithfulness",
  "answer_relevancy",
  "context_precision",
  "context_recall",
  "retrieval_status",
  "answer_quality_status",
  "overall_status",
  "error",
  "ragas_error",
  "updated_at"
];

function doGet() {
  return jsonResponse_({
    ok: true,
    message: "RAG evaluation endpoint is ready."
  });
}

function doPost(e) {
  try {
    const payload = JSON.parse(
      (e && e.postData && e.postData.contents) || "{}"
    );

    const properties = PropertiesService.getScriptProperties();
    const expectedToken = properties.getProperty("API_TOKEN");

    if (!expectedToken || payload.token !== expectedToken) {
      return jsonResponse_({
        ok: false,
        error: "Unauthorized"
      });
    }

    const spreadsheetId = properties.getProperty("SPREADSHEET_ID");

    if (!spreadsheetId) {
      throw new Error("Missing SPREADSHEET_ID script property.");
    }

    if (!payload.run_id || !payload.row || !payload.row.case_id) {
      throw new Error("Missing run_id or row.case_id.");
    }

    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);

    let sheet = spreadsheet.getSheetByName(SHEET_NAME);

    if (!sheet) {
      sheet = spreadsheet.insertSheet(SHEET_NAME);
    }

    let headers = [];

    if (sheet.getLastRow() === 0) {
      sheet.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]);
      headers = HEADERS.slice();
    } else {
      headers = sheet
        .getRange(1, 1, 1, sheet.getLastColumn())
        .getValues()[0]
        .map(String);

      const missingHeaders = HEADERS.filter(
        header => !headers.includes(header)
      );

      if (missingHeaders.length > 0) {
        sheet
          .getRange(
            1,
            headers.length + 1,
            1,
            missingHeaders.length
          )
          .setValues([missingHeaders]);

        headers = headers.concat(missingHeaders);
      }
    }

    const allRows = sheet.getDataRange().getValues();
    const runIdIndex = headers.indexOf("run_id");
    const caseIdIndex = headers.indexOf("case_id");

    let targetRow = -1;

    // Update the existing case from this run rather than
    // creating a duplicate when RAGAS finishes.
    for (let i = 1; i < allRows.length; i++) {
      if (
        String(allRows[i][runIdIndex]) === String(payload.run_id) &&
        String(allRows[i][caseIdIndex]) === String(payload.row.case_id)
      ) {
        targetRow = i + 1;
        break;
      }
    }

    const record = Object.assign({}, payload.row, {
      run_id: payload.run_id,
      sync_status: payload.sync_status || "updated",
      updated_at: new Date().toISOString()
    });

    const outputRow = headers.map(header =>
      toCellValue_(record[header])
    );

    if (targetRow > 0) {
      sheet
        .getRange(targetRow, 1, 1, headers.length)
        .setValues([outputRow]);
    } else {
      sheet.appendRow(outputRow);
    }

    SpreadsheetApp.flush();

    return jsonResponse_({
      ok: true,
      case_id: payload.row.case_id,
      action: targetRow > 0 ? "updated" : "inserted"
    });

  } catch (error) {
    return jsonResponse_({
      ok: false,
      error: String(error)
    });
  }
}

function toCellValue_(value) {
  if (value === null || value === undefined) {
    return "";
  }

  if (typeof value === "number" && !Number.isFinite(value)) {
    return "";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return value;
}

function jsonResponse_(data) {
  return ContentService
    .createTextOutput(JSON.stringify(data))
    .setMimeType(ContentService.MimeType.JSON);
}

function testSpreadsheetAccess() {
  const id = PropertiesService
    .getScriptProperties()
    .getProperty("SPREADSHEET_ID");

  const spreadsheet = SpreadsheetApp.openById(id);

  Logger.log("Spreadsheet access OK: " + spreadsheet.getName());
}