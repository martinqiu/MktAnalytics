args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)

if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = FALSE)
  project_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = FALSE)
} else {
  project_root <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}

setwd(project_root)

required_packages <- c("shiny", "bslib", "plotly", "DT")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing_packages) > 0) {
  stop(
    sprintf(
      "Missing required packages: %s. Run install_dependencies.bat first.",
      paste(missing_packages, collapse = ", ")
    ),
    call. = FALSE
  )
}

app_file <- file.path(project_root, "planning_dashboard", "app.R")
shiny::runApp(app_file, launch.browser = TRUE)
