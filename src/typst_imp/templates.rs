pub fn raw(code: &str) -> String {
    format!(
        r#"
#set page(height: auto, width: auto, margin: 0pt, fill: none)
#set text(16pt)
{code}
"#,
    )
}
