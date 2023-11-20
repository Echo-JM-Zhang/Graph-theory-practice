setwd("D:/JingminZhang/rutsuko's lab/DataAnalysis")

extract_cor <- function( tem_data){
  data <- tem_data[, 2:dim(tem_data)[2]]
  data[lower.tri(data, diag = TRUE)] <- NA
  cor_data <- c()
  for (i in 1:dim(data)[1]){
    for (j in 1:dim(data)[2]){
      if (!(is.na(data[i,j]))){
        cor_data <- c(cor_data, data[i,j])
      }
    }
  }
  return(cor_data)
}

# get unadjusted p values and assign orders
get_unadjusted_p <- function(cor_data, n){
  test_sta <- cor_data*sqrt(n-2)/sqrt(1-cor_data^2)
  test_sta <- pt(test_sta, df = n-2, lower.tail=FALSE)
  test_sta <- cbind(cor_data, test_sta)
  rank <- test_sta[order(test_sta[,2], decreasing = FALSE),]
  return(rank)
}


filename <- "Corr_APPROACH_TOP25.csv"
outputName <- "FDR_corrected_APPROACH_TOP25.csv"
mydata <- read.csv(filename, header = TRUE)


cor_vec <- extract_cor(mydata)
n = 10 # number of animals in this group
test <- get_unadjusted_p(cor_vec, n)

m <- dim(test)[1] # m is the total comparison number

adj_p <- p.adjust(test[,2], method="BH")
test <- cbind(test, adj_p)

write.csv(data.frame(test), file = outputName)
